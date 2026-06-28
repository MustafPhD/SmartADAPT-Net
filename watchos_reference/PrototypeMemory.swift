import Foundation

public struct PrototypeMemoryState: Codable, Sendable {
    public let participantID: String
    public let prototypes: [[Double]?]
    public let counts: [Int]
}

public final class PrototypeMemory {
    public let participantID: String
    public let nClasses: Int
    public let embeddingDim: Int
    public let nMin: Int
    public let lambda: Double
    public let alphaMin: Double
    public let alphaMax: Double

    private(set) public var prototypes: [[Double]?]
    private(set) public var counts: [Int]

    public init(
        participantID: String,
        nClasses: Int = 6,
        embeddingDim: Int = 128,
        nMin: Int = 3,
        lambda: Double = 10.0,
        alphaMin: Double = 0.05,
        alphaMax: Double = 0.30
    ) {
        self.participantID = participantID
        self.nClasses = nClasses
        self.embeddingDim = embeddingDim
        self.nMin = nMin
        self.lambda = lambda
        self.alphaMin = alphaMin
        self.alphaMax = alphaMax
        self.prototypes = Array(repeating: nil, count: nClasses)
        self.counts = Array(repeating: 0, count: nClasses)
    }

    public convenience init(state: PrototypeMemoryState, nMin: Int = 3, lambda: Double = 10.0, alphaMin: Double = 0.05, alphaMax: Double = 0.30) {
        let dim = state.prototypes.compactMap { $0?.count }.first ?? 128
        self.init(
            participantID: state.participantID,
            nClasses: state.counts.count,
            embeddingDim: dim,
            nMin: nMin,
            lambda: lambda,
            alphaMin: alphaMin,
            alphaMax: alphaMax
        )
        self.prototypes = state.prototypes
        self.counts = state.counts
    }

    public var state: PrototypeMemoryState {
        PrototypeMemoryState(participantID: participantID, prototypes: prototypes, counts: counts)
    }

    public var activeMask: [Bool] {
        counts.enumerated().map { index, count in
            count >= nMin && prototypes[index] != nil
        }
    }

    public var hasActivePrototype: Bool {
        activeMask.contains(true)
    }

    public func prototypeProbabilities(for embeddingL2: [Double]) -> [Double] {
        let z = ProbabilityMath.l2Normalize(embeddingL2)
        let active = activeMask

        guard active.contains(true) else {
            return Array(repeating: 1.0 / Double(nClasses), count: nClasses)
        }

        var logits = Array(repeating: -Double.infinity, count: nClasses)
        for classIndex in 0..<nClasses where active[classIndex] {
            guard let prototype = prototypes[classIndex] else { continue }
            logits[classIndex] = lambda * ProbabilityMath.dot(z, prototype)
        }

        let finite = logits.filter { $0.isFinite }
        let maxFinite = finite.max() ?? 0.0
        let exps = logits.map { $0.isFinite ? Foundation.exp($0 - maxFinite) : 0.0 }
        let denom = exps.reduce(0.0, +)
        guard denom > 0.0 else {
            return Array(repeating: 1.0 / Double(nClasses), count: nClasses)
        }
        return exps.map { $0 / denom }
    }

    public func update(confirmedClassIndex: Int, sessionEmbeddingL2: [Double], sessionComplexity: Double) {
        guard (0..<nClasses).contains(confirmedClassIndex) else { return }
        guard sessionEmbeddingL2.count == embeddingDim else { return }

        let z = ProbabilityMath.l2Normalize(sessionEmbeddingL2)
        let c = ProbabilityMath.clipped(sessionComplexity)
        let alpha = alphaMin + (alphaMax - alphaMin) * (1.0 - c)

        if let old = prototypes[confirmedClassIndex] {
            let updated = zip(old, z).map { (1.0 - alpha) * $0.0 + alpha * $0.1 }
            prototypes[confirmedClassIndex] = ProbabilityMath.l2Normalize(updated)
        } else {
            prototypes[confirmedClassIndex] = z
        }

        counts[confirmedClassIndex] += 1
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder.pretty.encode(state)
        try data.write(to: url, options: [.atomic])
    }

    public static func load(from url: URL) throws -> PrototypeMemory {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(PrototypeMemoryState.self, from: data)
        return PrototypeMemory(state: state)
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
