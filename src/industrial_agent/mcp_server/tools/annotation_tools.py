"""Annotation and label processing tools for MCP server.

This module provides tools for processing JSON annotations, converting formats,
and managing image datasets.
"""

import json
import os
import shutil
from collections import Counter
from glob import glob
from typing import Any, Optional

import cv2


class AnnotationProcessor:
    """Processor for annotation and label management."""

    def count_json_labels(self, directory: str, pattern: str = "*.json") -> dict[str, Any]:
        """
        Count labels/categories in JSON annotation files.

        Args:
            directory: Directory containing JSON files.
            pattern: File pattern to match (default: "*.json").

        Returns:
            Dictionary containing label statistics.
        """
        try:
            json_list = glob(os.path.join(directory, pattern))

            if not json_list:
                return {"error": f"No JSON files found in {directory}"}

            global_label_counts = Counter()
            global_unique_labels = set()

            for json_file in json_list:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract categories from shapes
                    if "shapes" in data:
                        categories = [item.get("label", "") for item in data["shapes"]]
                        global_label_counts.update(categories)
                        global_unique_labels.update(categories)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue

            return {
                "total_files": len(json_list),
                "total_categories": len(global_unique_labels),
                "categories": list(global_unique_labels),
                "label_counts": dict(global_label_counts),
            }

        except Exception as e:
            return {"error": f"Failed to count labels: {str(e)}"}

    def check_and_fix_image_paths(
        self,
        directory: str,
        image_extensions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Check and update imagePath in JSON files to match actual image files.

        Args:
            directory: Directory containing JSON and image files.
            image_extensions: List of supported image extensions (default: ['.jpg', '.jpeg', '.bmp', '.png']).

        Returns:
            Dictionary containing processing results.
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".bmp", ".png"]

        try:
            all_files = os.listdir(directory)

            # Build image mapping: base filename -> image filename
            image_mapping = {}
            for f in all_files:
                ext = os.path.splitext(f)[1].lower()
                if ext in image_extensions:
                    base = os.path.splitext(f)[0]
                    if base not in image_mapping:
                        image_mapping[base] = f

            # Process JSON files
            json_files = [f for f in all_files if f.endswith(".json")]
            updated_count = 0
            not_found = []

            for json_file in json_files:
                json_path = os.path.join(directory, json_file)

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                image_path = data.get("imagePath", "")
                image_name_without_ext = os.path.splitext(image_path)[0]

                if image_name_without_ext in image_mapping:
                    updated_image_path = image_mapping[image_name_without_ext]
                    if image_path != updated_image_path:
                        data["imagePath"] = updated_image_path

                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)

                        updated_count += 1
                else:
                    not_found.append(json_file)

            return {
                "total_json_files": len(json_files),
                "updated_files": updated_count,
                "not_found_images": not_found,
                "status": "success",
            }

        except Exception as e:
            return {"error": f"Failed to check image paths: {str(e)}"}

    def convert_labels_to_ng(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        label_mapping: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Convert Chinese labels to unified 'ng' (no good) labels.

        Args:
            input_dir: Directory containing input JSON files.
            output_dir: Output directory (default: same as input_dir).
            label_mapping: Dictionary mapping source labels to target labels.

        Returns:
            Dictionary containing conversion results.
        """
        if output_dir is None:
            output_dir = input_dir

        if label_mapping is None:
            # Default Chinese defect labels to 'ng'
            label_mapping = {
                "断胶": "ng",
                "残胶污迹": "ng",
                "异物混入": "ng",
                "接头缺胶": "ng",
                "胶分层": "ng",
                "记号笔印": "ng",
                "收胶不良": "ng",
                "胶宽超宽": "ng",
                "气泡": "ng",
                "透光": "ng",
                "胶宽不足": "ng",
                "cut assy": "ng",
            }

        try:
            os.makedirs(output_dir, exist_ok=True)
            json_files = glob(os.path.join(input_dir, "*.json"))

            converted_count = 0
            conversion_stats = Counter()

            for json_path in json_files:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Replace labels in shapes
                for shape in data.get("shapes", []):
                    original_label = shape.get("label", "")
                    if original_label in label_mapping:
                        shape["label"] = label_mapping[original_label]
                        conversion_stats[original_label] += 1

                # Save modified JSON
                new_json_path = os.path.join(output_dir, os.path.basename(json_path))
                with open(new_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                converted_count += 1

            return {
                "total_files": len(json_files),
                "converted_files": converted_count,
                "conversion_stats": dict(conversion_stats),
                "status": "success",
            }

        except Exception as e:
            return {"error": f"Failed to convert labels: {str(e)}"}

    def convert_to_coco_format(
        self,
        source_dir: str,
        output_file: str,
        label_mapping: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Convert LabelMe JSON format to COCO format.

        Args:
            source_dir: Directory containing source JSON files.
            output_file: Output COCO format JSON file path.
            label_mapping: Optional label mapping dictionary.

        Returns:
            Dictionary containing conversion results.
        """
        if label_mapping is None:
            label_mapping = {
                "断胶": "ng",
                "残胶污迹": "ng",
                "异物混入": "ng",
                "接头缺胶": "ng",
                "胶分层": "ng",
                "记号笔印": "ng",
                "收胶不良": "ng",
                "胶宽超宽": "ng",
                "气泡": "ng",
                "透光": "ng",
                "胶宽不足": "ng",
                "cut assy": "ng",
            }

        try:
            # Initialize COCO format
            coco_format = {
                "info": {
                    "description": "Converted Dataset",
                    "url": "",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "Industrial Agent",
                    "date_created": "2024-01-01",
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "CC-BY",
                        "url": "https://creativecommons.org/licenses/by/4.0/",
                    }
                ],
                "images": [],
                "annotations": [],
                "categories": [],
            }

            image_id = 0
            annotation_id = 0
            category_dict = {}

            # Process all JSON files
            json_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

            for filename in json_files:
                with open(os.path.join(source_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Add image info
                image_id += 1
                image_info = {
                    "id": image_id,
                    "file_name": data.get("imagePath", ""),
                    "width": data.get("imageWidth", 0),
                    "height": data.get("imageHeight", 0),
                }
                coco_format["images"].append(image_info)

                # Process annotations
                for shape in data.get("shapes", []):
                    category_name = shape.get("label", "unknown")
                    category_name = label_mapping.get(category_name, category_name)

                    # Add category if not exists
                    if category_name not in category_dict:
                        category_id = len(category_dict)
                        category_dict[category_name] = category_id
                        category_info = {
                            "id": category_id,
                            "name": category_name,
                            "supercategory": "defect",
                        }
                        coco_format["categories"].append(category_info)

                    category_id = category_dict[category_name]

                    if shape.get("shape_type") == "rectangle":
                        points = shape.get("points", [])
                        if len(points) >= 2:
                            x_min = min(point[0] for point in points)
                            y_min = min(point[1] for point in points)
                            width = max(point[0] for point in points) - x_min
                            height = max(point[1] for point in points) - y_min

                            annotation = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [x_min, y_min, width, height],
                                "area": width * height,
                                "segmentation": [],
                                "iscrowd": 0,
                            }
                            coco_format["annotations"].append(annotation)
                            annotation_id += 1

            # Save COCO format JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(coco_format, f, indent=4, ensure_ascii=False)

            return {
                "total_images": image_id,
                "total_annotations": annotation_id,
                "categories": list(category_dict.keys()),
                "output_file": output_file,
                "status": "success",
            }

        except Exception as e:
            return {"error": f"Failed to convert to COCO format: {str(e)}"}

    def split_dataset(
        self,
        source_dir: str,
        train_ratio: float = 0.8,
        image_extension: str = ".png",
    ) -> dict[str, Any]:
        """
        Split dataset into train and validation sets.

        Args:
            source_dir: Source directory containing images and JSON files.
            train_ratio: Ratio for training set (default: 0.8).
            image_extension: Image file extension to match (default: ".png").

        Returns:
            Dictionary containing split results.
        """
        try:
            import random

            train_dir = os.path.join(source_dir, "train")
            val_dir = os.path.join(source_dir, "val")

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Get all image files
            image_files = [f for f in os.listdir(source_dir) if f.endswith(image_extension)]

            train_count = 0
            val_count = 0

            for img_file in image_files:
                json_file = img_file.replace(image_extension, ".json")

                # Check if corresponding JSON exists
                if json_file in os.listdir(source_dir):
                    target_dir = train_dir if random.random() < train_ratio else val_dir

                    # Copy files
                    shutil.copy(
                        os.path.join(source_dir, img_file),
                        os.path.join(target_dir, img_file),
                    )
                    shutil.copy(
                        os.path.join(source_dir, json_file),
                        os.path.join(target_dir, json_file),
                    )

                    if target_dir == train_dir:
                        train_count += 1
                    else:
                        val_count += 1

            return {
                "total_files": len(image_files),
                "train_count": train_count,
                "val_count": val_count,
                "train_dir": train_dir,
                "val_dir": val_dir,
                "status": "success",
            }

        except Exception as e:
            return {"error": f"Failed to split dataset: {str(e)}"}

    def resize_dataset(
        self,
        input_dir: str,
        output_dir: str,
        scale: float = 0.5,
        label_mapping: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Resize images and adjust annotation coordinates.

        Args:
            input_dir: Input directory containing images and JSON files.
            output_dir: Output directory for resized data.
            scale: Scale factor for resizing (default: 0.5).
            label_mapping: Optional label mapping dictionary.

        Returns:
            Dictionary containing resize results.
        """
        if label_mapping is None:
            label_mapping = {
                "断胶": "ng",
                "残胶污迹": "ng",
                "异物混入": "ng",
                "接头缺胶": "ng",
                "胶分层": "ng",
                "记号笔印": "ng",
                "收胶不良": "ng",
                "胶宽超宽": "ng",
                "气泡": "ng",
                "透光": "ng",
                "胶宽不足": "ng",
                "cut assy": "ng",
            }

        try:
            os.makedirs(output_dir, exist_ok=True)
            json_files = glob(os.path.join(input_dir, "*.json"))

            processed_count = 0
            errors = []

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    img_name = data.get("imagePath", "")
                    img_path = os.path.join(input_dir, img_name)

                    if not os.path.exists(img_path):
                        errors.append(f"Image not found: {img_path}")
                        continue

                    # Read and resize image
                    img = cv2.imread(img_path)
                    if img is None:
                        errors.append(f"Failed to read image: {img_path}")
                        continue

                    h, w = img.shape[:2]
                    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))

                    # Save resized image
                    new_img_name = os.path.splitext(img_name)[0] + "_resized.jpg"
                    new_img_path = os.path.join(output_dir, new_img_name)
                    cv2.imwrite(new_img_path, resized_img)

                    # Update JSON
                    data["imageWidth"] = int(w * scale)
                    data["imageHeight"] = int(h * scale)
                    data["imagePath"] = new_img_name

                    for shape in data.get("shapes", []):
                        if shape.get("label") in label_mapping:
                            shape["label"] = label_mapping[shape["label"]]
                        shape["points"] = [[x * scale, y * scale] for x, y in shape.get("points", [])]

                    # Save new JSON
                    new_json_name = os.path.splitext(os.path.basename(json_file))[0] + "_resized.json"
                    new_json_path = os.path.join(output_dir, new_json_name)
                    with open(new_json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                    processed_count += 1

                except Exception as e:
                    errors.append(f"Error processing {json_file}: {str(e)}")

            return {
                "total_files": len(json_files),
                "processed_count": processed_count,
                "errors": errors,
                "output_dir": output_dir,
                "status": "success" if processed_count > 0 else "partial_success",
            }

        except Exception as e:
            return {"error": f"Failed to resize dataset: {str(e)}"}


# Global processor instance
_processor: Optional[AnnotationProcessor] = None


def get_annotation_processor() -> AnnotationProcessor:
    """Get or create the global annotation processor instance."""
    global _processor
    if _processor is None:
        _processor = AnnotationProcessor()
    return _processor
