import UIKit
import MapKit

class PlantAnnotationView: MKAnnotationView {

    static let reuseID = "PlantAnnotationView"

    private let pinSize: CGFloat = 44
    private let iconSize: CGFloat = 20
    private let orangeColor = UIColor(red: 1.0, green: 0.58, blue: 0.0, alpha: 1.0)

    // MARK: - Init

    override init(annotation: MKAnnotation?, reuseIdentifier: String?) {
        super.init(annotation: annotation, reuseIdentifier: reuseIdentifier)
        setupView()
    }

    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        setupView()
    }

    // MARK: - Setup

    private func setupView() {
        frame = CGRect(x: 0, y: 0, width: pinSize, height: pinSize)
        centerOffset = CGPoint(x: 0, y: -pinSize / 2)
        canShowCallout = false

        backgroundColor = orangeColor
        layer.cornerRadius = pinSize / 2
        layer.masksToBounds = false

        // Drop shadow
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.3
        layer.shadowRadius = 4
        layer.shadowOffset = CGSize(width: 0, height: 2)

        // Leaf icon
        let config = UIImage.SymbolConfiguration(pointSize: iconSize, weight: .semibold)
        let leafImage = UIImage(systemName: "leaf.fill", withConfiguration: config)
        let imageView = UIImageView(image: leafImage)
        imageView.tintColor = .white
        imageView.contentMode = .scaleAspectFit
        imageView.frame = CGRect(
            x: (pinSize - iconSize) / 2,
            y: (pinSize - iconSize) / 2,
            width: iconSize,
            height: iconSize
        )
        addSubview(imageView)
    }

    // MARK: - Selection

    override var isSelected: Bool {
        didSet {
            if isSelected {
                animateSelect()
                addPulseRing()
            } else {
                animateDeselect()
                removePulseRings()
            }
        }
    }

    private func animateSelect() {
        UIView.animate(
            withDuration: 0.2,
            delay: 0,
            usingSpringWithDamping: 0.6,
            initialSpringVelocity: 0.8,
            options: [],
            animations: { self.transform = CGAffineTransform(scaleX: 1.3, y: 1.3) }
        )
    }

    private func animateDeselect() {
        UIView.animate(
            withDuration: 0.2,
            delay: 0,
            usingSpringWithDamping: 0.6,
            initialSpringVelocity: 0.8,
            options: [],
            animations: { self.transform = .identity }
        )
    }

    // MARK: - Pulse Ring

    private func addPulseRing() {
        let pulse = CALayer()
        pulse.name = "pulseRing"
        pulse.frame = bounds
        pulse.cornerRadius = pinSize / 2
        pulse.borderWidth = 2
        pulse.borderColor = orangeColor.withAlphaComponent(0.6).cgColor
        pulse.backgroundColor = UIColor.clear.cgColor

        let scaleAnim = CABasicAnimation(keyPath: "transform.scale")
        scaleAnim.fromValue = 1.0
        scaleAnim.toValue = 1.9

        let opacityAnim = CABasicAnimation(keyPath: "opacity")
        opacityAnim.fromValue = 0.6
        opacityAnim.toValue = 0.0

        let group = CAAnimationGroup()
        group.animations = [scaleAnim, opacityAnim]
        group.duration = 1.2
        group.repeatCount = .infinity
        group.timingFunction = CAMediaTimingFunction(name: .easeOut)

        pulse.add(group, forKey: "pulseGroup")
        layer.insertSublayer(pulse, at: 0)
    }

    private func removePulseRings() {
        layer.sublayers?
            .filter { $0.name == "pulseRing" }
            .forEach { $0.removeFromSuperlayer() }
    }

    // MARK: - Reuse

    override func prepareForReuse() {
        super.prepareForReuse()
        removePulseRings()
        transform = .identity
    }
}
