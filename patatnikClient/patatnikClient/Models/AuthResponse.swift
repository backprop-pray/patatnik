import Foundation

struct AuthResponse: Codable {
    let token: String
    let id: Int
    let email: String

    var user: User {
        User(id: id, username: "User", email: email)
    }
}
