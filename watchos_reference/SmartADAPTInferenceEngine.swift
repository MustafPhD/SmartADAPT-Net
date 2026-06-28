import Foundation
import CoreML

public struct SmartADAPTWindowOutput: Sendable {
    public let baseLogits: [Double]
    public let baseProbabilities: [Double]
    public let deltaLogits: [Double]
    public let globalLogits: [Double]
    public let globalProbabilities: [Double]
    public let embeddingL2: [Double]
    public let complexity: ComplexityResult
}

public final class SmartADAPTInferenceEngine {
    public enum EngineError: Error, LocalizedError {
        case modelNotFound(String)
        case invalidWindowLength(expected: Int, actual: Int)
        case missingOutput(String)
        case invalidMultiArray(String)

        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let name): return "Could not find compiled Core ML model named \(name)."
            case .invalidWindowLength(let expected, let actual): return "Expected \(expected) samples but received \(actual)."
            case .missingOutput(let name): return "Core ML output \(name) was not found."
            case .invalidMultiArray(let name): return "Core ML output \(name) could not be converted from MLMultiArray."
            }
        }
    }

    private enum IO {
        static let inputName = "motion_window_raw"
        static let baseLogits = "base_logits"
        static let deltaLogits = "delta_logits"
        static let embeddingL2 = "embedding_l2"
    }

    private let model: MLModel
    private let complexityScorer: ComplexityScorer
    private let windowSize: Int
    private let channelCount: Int

    public init(model: MLModel, complexityScorer: ComplexityScorer, windowSize: Int = 400, channelCount: Int = 6) {
        self.model = model
        self.complexityScorer = complexityScorer
        self.windowSize = windowSize
        self.channelCount = channelCount
    }

    public convenience init(
        compiledModelName: String,
        complexityScorer: ComplexityScorer,
        bundle: Bundle = .main,
        windowSize: Int = 400,
        channelCount: Int = 6
    ) throws {
        guard let url = bundle.url(forResource: compiledModelName, withExtension: "mlmodelc") else {
            throw EngineError.modelNotFound(compiledModelName)
        }
        let model = try MLModel(contentsOf: url)
        self.init(model: model, complexityScorer: complexityScorer, windowSize: windowSize, channelCount: channelCount)
    }

    public func predict(window: MotionWindow) throws -> SmartADAPTWindowOutput {
        guard window.count == windowSize else {
            throw EngineError.invalidWindowLength(expected: windowSize, actual: window.count)
        }

        let input = try makeInputArray(window: window)
        let provider = try MLDictionaryFeatureProvider(dictionary: [IO.inputName: input])
        let prediction = try model.prediction(from: provider)

        let baseLogits = try vector(prediction, name: IO.baseLogits)
        let deltaLogits = try vector(prediction, name: IO.deltaLogits)
        let embedding = try vector(prediction, name: IO.embeddingL2)
        let complexity = complexityScorer.score(window: window)

        let baseProbabilities = ProbabilityMath.softmax(baseLogits)
        let globalLogits = zip(baseLogits, deltaLogits).map { $0.0 + complexity.gate * $0.1 }
        let globalProbabilities = ProbabilityMath.softmax(globalLogits)

        return SmartADAPTWindowOutput(
            baseLogits: baseLogits,
            baseProbabilities: baseProbabilities,
            deltaLogits: deltaLogits,
            globalLogits: globalLogits,
            globalProbabilities: globalProbabilities,
            embeddingL2: ProbabilityMath.l2Normalize(embedding),
            complexity: complexity
        )
    }

    private func makeInputArray(window: MotionWindow) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: windowSize), NSNumber(value: channelCount)], dataType: .float32)
        let matrix = window.modelInputMatrix()

        for t in 0..<windowSize {
            for c in 0..<channelCount {
                array[[NSNumber(value: 0), NSNumber(value: t), NSNumber(value: c)]] = NSNumber(value: Float(matrix[t][c]))
            }
        }
        return array
    }

    private func vector(_ provider: MLFeatureProvider, name: String) throws -> [Double] {
        guard let value = provider.featureValue(for: name) else {
            throw EngineError.missingOutput(name)
        }
        guard let array = value.multiArrayValue else {
            throw EngineError.invalidMultiArray(name)
        }
        return (0..<array.count).map { Double(truncating: array[$0]) }
    }
}
