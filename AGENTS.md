# AGENTS.md

You are a coding agent in a 5 person hackaton project.

## What this project is

Hackathon prototype for a low-cost smart farming rover.

The product idea is a small rover that moves between crops, uses cameras and sensors to detect plant problems, attaches a location, and sends the result to a mobile app so the farmer can act quickly.

## Main parts of the project

- **Rover / embedded code**  
  Reads cameras and sensors, moves through the field, detects issues, and reports findings.

- **Mobile app**  
  Shows fields and map locations, displays photos of detected problems, and lets the farmer confirm or correct the AI result.

- **Backend / cloud / gateway**  
  Stores telemetry, images, detections, and farmer feedback. Connects rover and app.

- **AI for anomaly detection**  
  Detects suspicious or unhealthy plants from images.

- **AI for navigation**  
  Helps the rover move safely between crops and around obstacles.

- **Knowledge / recommendation layer**  
  Turns detections and farmer feedback into better future suggestions.

