import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authViewModel: AuthViewModel

    var body: some View {
        Group {
            if authViewModel.isAuthenticated {
                HomeView()
            } else {
                NavigationStack {
                    LoginView()
                }
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AuthViewModel())
}
