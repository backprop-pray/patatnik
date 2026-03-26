package com.patatnik.server.controller;

import com.patatnik.server.exception.BadRequestException;
import com.patatnik.server.exception.ResourceNotFoundException;
import com.patatnik.server.jwt.JwtUtils;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.UserRepository;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public abstract class BaseController {

    protected final JwtUtils jwtUtils;
    protected final UserRepository userRepository;

    protected User resolveUser(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            throw new BadRequestException("Missing or invalid Authorization header");
        }

        String token = authHeader.substring(7);

        if (!jwtUtils.validate(token)) {
            throw new BadRequestException("Invalid or expired JWT token");
        }

        String email = jwtUtils.getEmailFromToken(token);

        return userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
    }
}
