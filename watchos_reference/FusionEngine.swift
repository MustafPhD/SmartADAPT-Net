import Foundation

public enum ProbabilityMath {
    public static func softmax(_ logits: [Double], temperature: Double = 1.0) -> [Double] {
        guard !logits.isEmpty else { return [] }
        let scaled = logits.map { $0 / max(temperature, 1e-12) }
        let m = scaled.max() ?? 0.0
        let exps = scaled.map { Foundation.exp($0 - m) }
        let z = exps.reduce(0.0, +)
        guard z > 0.0, z.isFinite else {
            return Array(repeating: 1.0 / Double(logits.count), count: logits.count)
        }
        return exps.map { $0 / z }
    }

    public static func sigmoid(_ x: Double) -> Double {
        if x >= 0.0 {
            let z = Foundation.exp(-x)
            return 1.0 / (1.0 + z)
        } else {
            let z = Foundation.exp(x)
            return z / (1.0 + z)
        }
    }

    public static func argmax(_ values: [Double]) -> Int {
        values.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    public static func l2Normalize(_ vector: [Double], epsilon: Double = 1e-12) -> [Double] {
        let norm = Foundation.sqrt(vector.reduce(0.0) { $0 + $1 * $1 })
        let denom = max(norm, epsilon)
        return vector.map { $0 / denom }
    }

    public static func dot(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
    }

    public static func mean(_ vectors: [[Double]]) -> [Double] {
        guard let first = vectors.first else { return [] }
        var out = Array(repeating: 0.0, count: first.count)
        for vector in vectors {
            for i in 0..<out.count { out[i] += vector[i] }
        }
        let n = Double(vectors.count)
        return out.map { $0 / n }
    }

    public static func weightedMean(vectors: [[Double]], weights: [Double]) -> [Double] {
        guard let first = vectors.first, vectors.count == weights.count else { return [] }
        var out = Array(repeating: 0.0, count: first.count)
        let denom = max(weights.reduce(0.0, +), 1e-12)
        for (vector, weight) in zip(vectors, weights) {
            for i in 0..<out.count { out[i] += weight * vector[i] }
        }
        return out.map { $0 / denom }
    }

    public static func clipped(_ value: Double, lower: Double = 0.0, upper: Double = 1.0) -> Double {
        min(max(value, lower), upper)
    }
}

public struct FusionEngine: Sendable {
    public let gamma: Double
    public let nRef: Int

    public init(gamma: Double = 0.5, nRef: Int = 5) {
        self.gamma = gamma
        self.nRef = nRef
    }

    public func personalEvidenceRatio(counts: [Int], nClasses: Int) -> Double {
        guard nClasses > 0 else { return 0.0 }
        let ref = max(Double(nRef), 1.0)
        let coverage = counts.prefix(nClasses).map { min(Double($0) / ref, 1.0) }
        return coverage.reduce(0.0, +) / Double(nClasses)
    }

    public func beta(counts: [Int], nClasses: Int, hasActivePrototype: Bool) -> Double {
        guard hasActivePrototype else { return 1.0 }
        let rho = personalEvidenceRatio(counts: counts, nClasses: nClasses)
        return ProbabilityMath.clipped(1.0 - gamma * rho)
    }

    public func fuse(global: [Double], prototype: [Double], beta: Double) -> [Double] {
        precondition(global.count == prototype.count, "Global and prototype vectors must have the same class dimension.")
        let b = ProbabilityMath.clipped(beta)
        return zip(global, prototype).map { b * $0.0 + (1.0 - b) * $0.1 }
    }
}
