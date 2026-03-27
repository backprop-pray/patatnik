package com.patatnik.server.service.recomendation;

import com.patatnik.server.model.Plant;
import com.patatnik.server.model.User;
import lombok.RequiredArgsConstructor;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class RecommendationService {

    private final SimpMessagingTemplate messagingTemplate;
    private final RestTemplate restTemplate;

    @Value("${ai.service.url}")
    private String pythonUrl;

    @Async("recommendationExecutor")
    public void requestAnalysis(Plant plant, User user) {
        try {
            System.out.println("[AI] Requesting analysis for plant " + plant.getId());

            Map<String, Object> request = new HashMap<>();
            request.put("plant_id", plant.getId());
            request.put("image_url", plant.getImageUrl());

            if (plant.getImageUrl() == null || plant.getImageUrl().isBlank()) {
                System.out.println("[AI] Plant " + plant.getId() + " has no image, skipping");
                return;
            }

            Map response = restTemplate.postForObject(
                pythonUrl,
                request,
                Map.class
            );

            if (response == null) {
                System.out.println("[AI] Null response for plant " + plant.getId());
                return;
            }

            String text = (String) response.get("text");
            String disease = (String) response.getOrDefault("disease", "");

            System.out.println("[AI] Got recommendation for plant " 
                + plant.getId() + ": " + disease + " — " + text);

            // Push to user via WebSocket
            Map<String, Object> payload = new HashMap<>();
            payload.put("plantId", plant.getId());
            payload.put("disease", disease);
            payload.put("text", text);

            System.err.println(payload);

            messagingTemplate.convertAndSendToUser(
                user.getId().toString(),
                "/queue/recommendations",
                payload
            );

            System.out.println("[AI] Recommendation sent to user " + user.getId());

        } catch (Exception e) {
            System.out.println("[AI] Failed for plant " + plant.getId() + ": " + e.getMessage());
        }
    }
}