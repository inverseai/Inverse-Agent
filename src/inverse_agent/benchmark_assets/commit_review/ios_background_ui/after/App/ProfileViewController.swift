import UIKit

final class ProfileViewController: UIViewController {
    @IBOutlet private weak var statusLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        refreshStatus()
    }

    private func refreshStatus() {
        let url = URL(string: "https://status.example.test/profile")!
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let data else { return }
            self?.statusLabel.text = String(decoding: data, as: UTF8.self)
        }.resume()
    }
}
