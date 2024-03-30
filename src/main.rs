use anyhow::Result;
use itertools::izip;
use opencv::{core, dnn, highgui, imgcodecs, imgproc, prelude::*};

fn main() -> Result<()> {
    // Setup model based on YOLOv4 config and weights
    let config = "yolov4-tiny.cfg";
    let weights = "yolov4-tiny.weights";
    let net = dnn::read_net(weights, config, "Darknet")?;
    let mut model = dnn::DetectionModel::new_1(&net)?;
    let scale: f64 = 1.0 / 255.0;
    let size = core::Size {
        width: 416,
        height: 416,
    };
    let mean = core::Scalar {
        0: [0.0, 0.0, 0.0, 0.0], // Generally for YOLO
    };
    let swap_rb: bool = true;
    let crop: bool = false;
    model.set_input_params(scale, size, mean, swap_rb, crop)?;

    // Vecs to store detections
    let mut class_ids = core::Vector::<i32>::new();
    let mut confidences = core::Vector::<f32>::new();
    let mut boxes = core::Vector::<core::Rect>::new();

    // Open img
    let img_file = "img.jpg";
    let mut img = imgcodecs::imread_def(img_file)?;

    // Perform detections
    model.detect_def(&img, &mut class_ids, &mut confidences, &mut boxes)?;

    // Put bounding boxes on the img
    let color = core::Scalar {
        0: [0.0, 140.0, 255.0, 0.0], // Orange
    };
    for (_cid, _cf, b) in izip!(&class_ids, &confidences, &boxes) {
        imgproc::rectangle_def(&mut img, b, color)?;
    }

    // Display in a GUI window
    highgui::named_window("YOLOv4", highgui::WINDOW_FULLSCREEN)?;
    highgui::imshow("YOLOv4", &img)?;
    highgui::wait_key_def()?;

    Ok(())
}
