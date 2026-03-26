import SwiftUI

@main
struct patatnikClientApp: App {
    @StateObject private var authViewModel = AuthViewModel()
    @StateObject private var plantViewModel = PlantViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authViewModel)
                .environmentObject(plantViewModel)
        }
    }
}
