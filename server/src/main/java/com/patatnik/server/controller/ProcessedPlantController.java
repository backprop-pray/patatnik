package com.patatnik.server.controller;

import com.patatnik.server.dto.ApiResponse;
import com.patatnik.server.jwt.JwtUtils;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.UserRepository;
import com.patatnik.server.service.processedPlant.ProcessedPlantService;

import lombok.Getter;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/processed-plants")
public class ProcessedPlantController extends BaseController {

    private final ProcessedPlantService processedPlantService;

    public ProcessedPlantController(
            JwtUtils jwtUtils,
            UserRepository userRepository,
            ProcessedPlantService processedPlantService) {
        super(jwtUtils, userRepository);
        this.processedPlantService = processedPlantService;
    }

    @PostMapping("/{id}/accept")
    public ResponseEntity<ApiResponse<Void>> accept(
            @PathVariable Long id,
            @RequestHeader("Authorization") String authHeader) {
        User user = resolveUser(authHeader);
        processedPlantService.accept(id, user);
        return ResponseEntity.ok(ApiResponse.success(200, "Accepted", null));
    }

    @PostMapping("/{id}/reject")
    public ResponseEntity<ApiResponse<Void>> reject(
            @PathVariable Long id,
            @RequestBody RejectRequest body,
            @RequestHeader("Authorization") String authHeader) {
        User user = resolveUser(authHeader);
        processedPlantService.reject(id, body.getComment(), user);
        return ResponseEntity.ok(ApiResponse.success(200, "Rejected", null));
    }

    @Getter
    static class RejectRequest {
        private String comment;
    }
}