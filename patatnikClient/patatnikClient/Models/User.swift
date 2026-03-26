import Foundation

nonisolated struct User: Codable, Identifiable, Sendable {
    let id: Int
    let username: String
    let email: String
}
