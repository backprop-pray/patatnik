import SwiftUI
import MapKit

struct MapView: UIViewRepresentable {
    let plants: [Plant]
    let mapType: MKMapType
    @Binding var selectedPlant: Plant?

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeUIView(context: Context) -> MKMapView {
        let mapView = MKMapView()
        mapView.delegate = context.coordinator
        mapView.mapType = .hybridFlyover
        mapView.showsUserLocation = true
        mapView.showsCompass = true
        mapView.showsScale = true
        mapView.showsBuildings = true
        mapView.isPitchEnabled = true

        mapView.register(
            PlantAnnotationView.self,
            forAnnotationViewWithReuseIdentifier: MKMapViewDefaultAnnotationViewReuseIdentifier
        )

        return mapView
    }

    func updateUIView(_ mapView: MKMapView, context: Context) {
        mapView.mapType = mapType

        // Sync annotations
        let existing = mapView.annotations.filter { !($0 is MKUserLocation) }
        mapView.removeAnnotations(existing)

        let newAnnotations = plants.map { PlantAnnotation(plant: $0) }
        mapView.addAnnotations(newAnnotations)

        // Auto-fit on first load (0 → N)
        let wasEmpty = context.coordinator.previousPlantCount == 0
        context.coordinator.previousPlantCount = plants.count

        if wasEmpty && !plants.isEmpty {
            var rect = MKMapRect.null
            for annotation in newAnnotations {
                let point = MKMapPoint(annotation.coordinate)
                let pointRect = MKMapRect(x: point.x, y: point.y, width: 0.1, height: 0.1)
                rect = rect.union(pointRect)
            }
            mapView.setVisibleMapRect(
                rect,
                edgePadding: UIEdgeInsets(top: 80, left: 40, bottom: 120, right: 40),
                animated: true
            )
        }
    }

    // MARK: - Coordinator

    class Coordinator: NSObject, MKMapViewDelegate {
        let parent: MapView
        var previousPlantCount = 0

        init(parent: MapView) {
            self.parent = parent
        }

        func mapView(_ mapView: MKMapView, viewFor annotation: MKAnnotation) -> MKAnnotationView? {
            if annotation is MKUserLocation { return nil }

            let view = mapView.dequeueReusableAnnotationView(
                withIdentifier: MKMapViewDefaultAnnotationViewReuseIdentifier,
                for: annotation
            )
            return view
        }

        func mapView(_ mapView: MKMapView, didSelect view: MKAnnotationView) {
            guard let plantAnnotation = view.annotation as? PlantAnnotation else { return }
            parent.selectedPlant = plantAnnotation.plant
            print("Selected plant id:", plantAnnotation.plant.id)
        }

        func mapView(_ mapView: MKMapView, didDeselect view: MKAnnotationView) {
            // Do not clear selectedPlant — popup close button handles dismissal
        }
    }
}
