import Foundation

public struct ComplexityCalibration: Codable, Sendable {
    public let descriptorMean: [Double]
    public let descriptorStd: [Double]
    public let q15: Double
    public let q85: Double
    public let wAcc: Double
    public let tau: Double
    public let gateK: Double

    public init(
        descriptorMean: [Double],
        descriptorStd: [Double],
        q15: Double,
        q85: Double,
        wAcc: Double = 0.4,
        tau: Double = 0.5,
        gateK: Double = 10.0
    ) {
        precondition(descriptorMean.count == 6, "descriptorMean must contain six values.")
        precondition(descriptorStd.count == 6, "descriptorStd must contain six values.")
        self.descriptorMean = descriptorMean
        self.descriptorStd = descriptorStd
        self.q15 = q15
        self.q85 = q85
        self.wAcc = wAcc
        self.tau = tau
        self.gateK = gateK
    }
}

public struct ComplexityResult: Codable, Sendable {
    public let descriptorVector: [Double]
    public let raw: Double
    public let calibrated: Double
    public let gate: Double
}

public struct ComplexityScorer: Sendable {
    public let calibration: ComplexityCalibration

    public init(calibration: ComplexityCalibration) {
        self.calibration = calibration
    }

    public func score(window: MotionWindow) -> ComplexityResult {
        let acc = window.accelerationMatrix()
        let gyro = window.rotationRateMatrix()

        let a = descriptors(acc)
        let g = descriptors(gyro)
        let descriptorVector = [a.absMean, a.rangeMean, a.diff2Mean, g.absMean, g.rangeMean, g.diff2Mean]

        let z = zip(descriptorVector, zip(calibration.descriptorMean, calibration.descriptorStd)).map { value, stats in
            let sd = max(stats.1, 1e-12)
            return (value - stats.0) / sd
        }

        let cAcc = (z[0] + z[1] + z[2]) / 3.0
        let cGyro = (z[3] + z[4] + z[5]) / 3.0
        let raw = calibration.wAcc * cAcc + (1.0 - calibration.wAcc) * cGyro

        let denom = max(calibration.q85 - calibration.q15, 1e-12)
        let calibrated = ProbabilityMath.clipped((raw - calibration.q15) / denom)
        let gate = ProbabilityMath.sigmoid(calibration.gateK * (calibrated - calibration.tau))

        return ComplexityResult(
            descriptorVector: descriptorVector,
            raw: raw,
            calibrated: calibrated,
            gate: gate
        )
    }

    private struct DescriptorTriplet {
        let absMean: Double
        let rangeMean: Double
        let diff2Mean: Double
    }

    private func descriptors(_ matrix: [[Double]]) -> DescriptorTriplet {
        guard !matrix.isEmpty else {
            return DescriptorTriplet(absMean: 0.0, rangeMean: 0.0, diff2Mean: 0.0)
        }

        let axes = 3
        let t = matrix.count
        var absSum = 0.0
        var minValues = Array(repeating: Double.greatestFiniteMagnitude, count: axes)
        var maxValues = Array(repeating: -Double.greatestFiniteMagnitude, count: axes)

        for row in matrix {
            for axis in 0..<axes {
                let value = row[axis]
                absSum += abs(value)
                minValues[axis] = min(minValues[axis], value)
                maxValues[axis] = max(maxValues[axis], value)
            }
        }

        let absMean = absSum / Double(max(t * axes, 1))
        let rangeMean = zip(maxValues, minValues).map { $0.0 - $0.1 }.reduce(0.0, +) / Double(axes)

        var diff2Sum = 0.0
        if t >= 3 {
            for i in 1..<(t - 1) {
                for axis in 0..<axes {
                    let v = matrix[i + 1][axis] - 2.0 * matrix[i][axis] + matrix[i - 1][axis]
                    diff2Sum += abs(v)
                }
            }
        }
        let diff2Denom = max((t - 2) * axes, 1)
        let diff2Mean = diff2Sum / Double(diff2Denom)

        return DescriptorTriplet(absMean: absMean, rangeMean: rangeMean, diff2Mean: diff2Mean)
    }
}
