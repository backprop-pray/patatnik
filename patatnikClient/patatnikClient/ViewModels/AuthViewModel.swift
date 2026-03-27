import SwiftUI
import Combine

@MainActor
class AuthViewModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var errorMessage: String?
    @Published var isLoading = false
    @Published private(set) var token: String?

    private let authService = AuthService()

    func login(email: String, password: String) async {
        errorMessage = nil
        isLoading = true
        defer { isLoading = false }

        do {
            let response = try await authService.login(email: email, password: password)
            self.token = response.token
            currentUser = response.user
            isAuthenticated = true
            
            // Clear any stale plant cache data
            clearPlantCache()
            
            PlantWebSocketService.shared.connect(userId: response.user.id)
        } catch let error as AuthService.AuthError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = "Something went wrong. Please try again."
        }
    }

    func register(name: String, email: String, password: String) async {
        errorMessage = nil
        isLoading = true
        defer { isLoading = false }

        do {
            let response = try await authService.register(name: name, email: email, password: password)
            self.token = response.token
            currentUser = response.user
            isAuthenticated = true
            
            // Clear any stale plant cache data
            clearPlantCache()
            
            PlantWebSocketService.shared.connect(userId: response.user.id)
        } catch let error as AuthService.AuthError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = "Something went wrong. Please try again."
        }
    }

    func logout() {
        PlantWebSocketService.shared.disconnect()
        token = nil
        currentUser = nil
        isAuthenticated = false
        
        // Clear plant cache on logout
        clearPlantCache()
    }
    
    private func clearPlantCache() {
        // Remove all plant-related UserDefaults cache
        let defaults = UserDefaults.standard
        let dictionary = defaults.dictionaryRepresentation()
        
        for key in dictionary.keys {
            if key.hasPrefix("plant_") {
                defaults.removeObject(forKey: key)
            }
        }
        
        print("[AuthVM] Cleared plant cache")
    }
}
