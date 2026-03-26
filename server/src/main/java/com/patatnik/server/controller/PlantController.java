package com.patatnik.server.controller;

import com.patatnik.server.dto.ApiResponse;
import com.patatnik.server.dto.PlantResponse;
import com.patatnik.server.jwt.JwtUtils;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.UserRepository;
import com.patatnik.server.service.plant.PlantService;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/plants")
public class PlantController extends BaseController {

    private final PlantService plantService;

    public PlantController(JwtUtils jwtUtils, UserRepository userRepository, PlantService plantService) {
        super(jwtUtils, userRepository);
        this.plantService = plantService;
    }

    @PostMapping
    public ResponseEntity<ApiResponse<PlantResponse>> createPlant(
            @RequestParam("latitude") Double latitude,
            @RequestParam("longitude") Double longitude,
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestHeader("Authorization") String authHeader) {

        User user = resolveUser(authHeader);

        PlantResponse data = plantService.createPlant(latitude, longitude, image, user);

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(ApiResponse.success(201, "Plant created successfully", data));
    }
}
