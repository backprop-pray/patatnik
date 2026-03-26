import Foundation

struct Plant: Codable, Identifiable {
    let id: Int
    let name: String
    let latitude: Double
    let longitude: Double
    let description: String?
    let imageURL: String?
    let userId: Int?
    let createdAt: String?
}
