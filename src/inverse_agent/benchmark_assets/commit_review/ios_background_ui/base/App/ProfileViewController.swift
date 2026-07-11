import UIKit

final class ProfileViewController: UIViewController {
    @IBOutlet private weak var statusLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        statusLabel.text = "Ready"
    }
}
