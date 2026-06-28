import Foundation

public struct WindowDecision: Sendable {
    public let baseProbabilities: [Double]
    public let globalProbabilities: [Double]
    public let prototypeProbabilities: [Double]
    public let finalProbabilities: [Double]
    public let embeddingL2: [Double]
    public let complexity: ComplexityResult
    public let beta: Double

    public var basePrediction: Int { ProbabilityMath.argmax(baseProbabilities) }
    public var globalPrediction: Int { ProbabilityMath.argmax(globalProbabilities) }
    public var prototypePrediction: Int { ProbabilityMath.argmax(prototypeProbabilities) }
    public var finalPrediction: Int { ProbabilityMath.argmax(finalProbabilities) }
}

public struct SessionPrediction: Sendable {
    public let participantID: String
    public let windowDecisions: [WindowDecision]
    public let baseProbabilities: [Double]
    public let globalProbabilities: [Double]
    public let prototypeProbabilities: [Double]
    public let finalProbabilities: [Double]
    public let sessionComplexity: Double

    public var basePrediction: Int { ProbabilityMath.argmax(baseProbabilities) }
    public var globalPrediction: Int { ProbabilityMath.argmax(globalProbabilities) }
    public var prototypePrediction: Int { ProbabilityMath.argmax(prototypeProbabilities) }
    public var finalPrediction: Int { ProbabilityMath.argmax(finalProbabilities) }
}

public final class SessionController {
    public enum SessionError: Error, LocalizedError {
        case insufficientSamples(required: Int, actual: Int)
        case invalidClassIndex(Int)
        case emptyWindowSet

        public var errorDescription: String? {
            switch self {
            case .insufficientSamples(let required, let actual):
                return "At least \(required) samples are required but only \(actual) were received."
            case .invalidClassIndex(let index):
                return "Invalid class index: \(index)."
            case .emptyWindowSet:
                return "No analysis windows could be constructed."
            }
        }
    }

    public let participantID: String
    public let sampleRateHz: Int
    public let preStabilisationSeconds: Int
    public let analysisSeconds: Int
    public let windowSeconds: Int
    public let nClasses: Int

    private let inferenceEngine: SmartADAPTInferenceEngine
    private let prototypeMemory: PrototypeMemory
    private let fusionEngine: FusionEngine

    public init(
        participantID: String,
        inferenceEngine: SmartADAPTInferenceEngine,
        prototypeMemory: PrototypeMemory,
        fusionEngine: FusionEngine = FusionEngine(gamma: 0.5, nRef: 5),
        sampleRateHz: Int = 50,
        preStabilisationSeconds: Int = 3,
        analysisSeconds: Int = 24,
        windowSeconds: Int = 8,
        nClasses: Int = 6
    ) {
        self.participantID = participantID
        self.inferenceEngine = inferenceEngine
        self.prototypeMemory = prototypeMemory
        self.fusionEngine = fusionEngine
        self.sampleRateHz = sampleRateHz
        self.preStabilisationSeconds = preStabilisationSeconds
        self.analysisSeconds = analysisSeconds
        self.windowSeconds = windowSeconds
        self.nClasses = nClasses
    }

    public func predictSession(samples: [MotionSample]) throws -> SessionPrediction {
        let windows = try makeAnalysisWindows(samples: samples)
        guard !windows.isEmpty else { throw SessionError.emptyWindowSet }

        var decisions: [WindowDecision] = []
        decisions.reserveCapacity(windows.count)

        for window in windows {
            let modelOutput = try inferenceEngine.predict(window: window)
            let proto = prototypeMemory.prototypeProbabilities(for: modelOutput.embeddingL2)
            let beta = fusionEngine.beta(
                counts: prototypeMemory.counts,
                nClasses: nClasses,
                hasActivePrototype: prototypeMemory.hasActivePrototype
            )
            let final = fusionEngine.fuse(global: modelOutput.globalProbabilities, prototype: proto, beta: beta)

            decisions.append(
                WindowDecision(
                    baseProbabilities: modelOutput.baseProbabilities,
                    globalProbabilities: modelOutput.globalProbabilities,
                    prototypeProbabilities: proto,
                    finalProbabilities: final,
                    embeddingL2: modelOutput.embeddingL2,
                    complexity: modelOutput.complexity,
                    beta: beta
                )
            )
        }

        return SessionPrediction(
            participantID: participantID,
            windowDecisions: decisions,
            baseProbabilities: ProbabilityMath.mean(decisions.map { $0.baseProbabilities }),
            globalProbabilities: ProbabilityMath.mean(decisions.map { $0.globalProbabilities }),
            prototypeProbabilities: ProbabilityMath.mean(decisions.map { $0.prototypeProbabilities }),
            finalProbabilities: ProbabilityMath.mean(decisions.map { $0.finalProbabilities }),
            sessionComplexity: decisions.map { $0.complexity.calibrated }.reduce(0.0, +) / Double(decisions.count)
        )
    }

    /// Call this only after the participant has confirmed or corrected the predicted class.
    public func applyConfirmation(_ prediction: SessionPrediction, confirmedClassIndex: Int) throws {
        guard (0..<nClasses).contains(confirmedClassIndex) else {
            throw SessionError.invalidClassIndex(confirmedClassIndex)
        }

        let embeddings = prediction.windowDecisions.map { $0.embeddingL2 }
        let weights = prediction.windowDecisions.map { $0.finalProbabilities[confirmedClassIndex] }
        let sessionEmbedding = ProbabilityMath.l2Normalize(
            ProbabilityMath.weightedMean(vectors: embeddings, weights: weights)
        )

        prototypeMemory.update(
            confirmedClassIndex: confirmedClassIndex,
            sessionEmbeddingL2: sessionEmbedding,
            sessionComplexity: prediction.sessionComplexity
        )
    }

    public func prototypeState() -> PrototypeMemoryState {
        prototypeMemory.state
    }

    private func makeAnalysisWindows(samples: [MotionSample]) throws -> [MotionWindow] {
        let preSamples = preStabilisationSeconds * sampleRateHz
        let analysisSamples = analysisSeconds * sampleRateHz
        let windowSamples = windowSeconds * sampleRateHz
        let required = preSamples + analysisSamples

        guard samples.count >= required else {
            throw SessionError.insufficientSamples(required: required, actual: samples.count)
        }

        let analysisStart = preSamples
        let analysisEnd = analysisStart + analysisSamples
        let analysisRegion = Array(samples[analysisStart..<analysisEnd])

        var windows: [MotionWindow] = []
        var start = 0
        while start + windowSamples <= analysisRegion.count {
            let end = start + windowSamples
            windows.append(MotionWindow(samples: Array(analysisRegion[start..<end])))
            start += windowSamples
        }

        return windows
    }
}
