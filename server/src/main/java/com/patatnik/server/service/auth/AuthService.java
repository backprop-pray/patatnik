package com.patatnik.server.service.auth;

import com.patatnik.server.dto.AuthResponse;
import com.patatnik.server.dto.LoginRequest;
import com.patatnik.server.dto.RegisterRequest;

public interface AuthService {

    AuthResponse register(RegisterRequest request);

    AuthResponse login(LoginRequest request);
}
