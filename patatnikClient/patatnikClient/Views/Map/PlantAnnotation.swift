import MapKit

class PlantAnnotation: MKPointAnnotation {
    let plant: Plant

    init(plant: Plant) {
        self.plant = plant
        super.init()
        self.coordinate = CLLocationCoordinate2D(
            latitude: plant.latitude,
            longitude: plant.longitude
        )
        self.title = "Plant #\(plant.id)"
    }
}
