def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    import numpy as np
    
    stride = image_size / feature_size
    anchor_boxes = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx, cy =  (j + 0.5) * stride, (i + 0.5) * stride
            for s in scales:
                for r in aspect_ratios:
                    w, h = s * np.sqrt(r), s / np.sqrt(r)
                    anchor_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return anchor_boxes