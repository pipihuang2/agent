"""YOLO training tools for MCP server."""

import json
import os
import threading
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ultralytics import YOLO


class YOLOTrainer:
    """YOLO model training manager supporting both sync and async training."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize YOLO trainer.

        Args:
            output_dir: Output directory for training results.
                       Default: ./output/yolo_training
        """
        self.output_dir = output_dir or Path("./output/yolo_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track async training status
        self._training_status: dict[str, dict] = {}
        self._training_threads: dict[str, threading.Thread] = {}

    # === Synchronous Training Methods ===

    def train_detection(
        self,
        data_yaml: str,
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: Optional[str] = None,
        name: Optional[str] = None,
        patience: int = 50,
        device: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """Train YOLO detection model (synchronous).

        Args:
            data_yaml: Path to YOLO data configuration file
            model_size: Model size (n/s/m/l/x), default "n"
            epochs: Number of training epochs, default 100
            imgsz: Image size for training, default 640
            batch: Batch size, default 16
            project: Project name for organizing experiments
            name: Experiment name
            patience: Early stopping patience, default 50
            device: Training device (auto/cpu/0/0,1), default auto
            pretrained: Use pretrained weights, default True
            **kwargs: Additional training parameters

        Returns:
            dict: Training results including model path, metrics, and logs
        """
        try:
            # Generate training ID
            training_id = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Setup project and name
            if project is None:
                project = str(self.output_dir)
            if name is None:
                name = training_id

            # Load model
            model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
            model = YOLO(model_name)

            # Train
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                patience=patience,
                device=device if device else "auto",
                **kwargs
            )

            # Format results
            return self._format_detection_results(results, training_id, project, name)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Training failed: {str(e)}"
            }

    def train_classification(
        self,
        data_dir: str,
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 224,
        batch: int = 32,
        project: Optional[str] = None,
        name: Optional[str] = None,
        patience: int = 50,
        device: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """Train YOLO classification model (synchronous).

        Args:
            data_dir: Path to dataset directory (containing train/ and val/ subdirs)
            model_size: Model size (n/s/m/l/x), default "n"
            epochs: Number of training epochs, default 100
            imgsz: Image size for training, default 224
            batch: Batch size, default 32
            project: Project name for organizing experiments
            name: Experiment name
            patience: Early stopping patience, default 50
            device: Training device (auto/cpu/0/0,1), default auto
            pretrained: Use pretrained weights, default True
            **kwargs: Additional training parameters

        Returns:
            dict: Training results including model path, metrics, and logs
        """
        try:
            # Generate training ID
            training_id = f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Setup project and name
            if project is None:
                project = str(self.output_dir)
            if name is None:
                name = training_id

            # Load classification model
            model_name = f"yolov8{model_size}-cls.pt" if pretrained else f"yolov8{model_size}-cls.yaml"
            model = YOLO(model_name)

            # Train
            results = model.train(
                data=data_dir,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                patience=patience,
                device=device if device else "auto",
                **kwargs
            )

            # Format results
            return self._format_classification_results(results, training_id, project, name)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Training failed: {str(e)}"
            }

    # === Asynchronous Training Methods ===

    def train_detection_async(
        self,
        data_yaml: str,
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: Optional[str] = None,
        name: Optional[str] = None,
        patience: int = 50,
        device: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """Start detection training in background (asynchronous).

        Args:
            Same as train_detection()

        Returns:
            dict: Training ID and status information
        """
        try:
            # Generate training ID
            training_id = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Setup project and name
            if project is None:
                project = str(self.output_dir)
            if name is None:
                name = training_id

            # Prepare training parameters
            params = {
                "data": data_yaml,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "project": project,
                "name": name,
                "patience": patience,
                "device": device if device else "auto",
                **kwargs
            }

            # Initialize status
            self._training_status[training_id] = {
                "status": "starting",
                "task_type": "detection",
                "start_time": datetime.now().isoformat(),
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": epochs,
                "params": params,
                "model_size": model_size,
                "pretrained": pretrained,
            }

            # Create and start background thread
            thread = threading.Thread(
                target=self._run_training_worker,
                args=(training_id, "detection", model_size, pretrained, params),
                daemon=True
            )
            thread.start()
            self._training_threads[training_id] = thread

            return {
                "success": True,
                "training_id": training_id,
                "status": "started",
                "message": "Training started in background. Use yolo_get_training_status to check progress."
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start training: {str(e)}"
            }

    def train_classification_async(
        self,
        data_dir: str,
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 224,
        batch: int = 32,
        project: Optional[str] = None,
        name: Optional[str] = None,
        patience: int = 50,
        device: Optional[str] = None,
        pretrained: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """Start classification training in background (asynchronous).

        Args:
            Same as train_classification()

        Returns:
            dict: Training ID and status information
        """
        try:
            # Generate training ID
            training_id = f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Setup project and name
            if project is None:
                project = str(self.output_dir)
            if name is None:
                name = training_id

            # Prepare training parameters
            params = {
                "data": data_dir,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "project": project,
                "name": name,
                "patience": patience,
                "device": device if device else "auto",
                **kwargs
            }

            # Initialize status
            self._training_status[training_id] = {
                "status": "starting",
                "task_type": "classification",
                "start_time": datetime.now().isoformat(),
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": epochs,
                "params": params,
                "model_size": model_size,
                "pretrained": pretrained,
            }

            # Create and start background thread
            thread = threading.Thread(
                target=self._run_training_worker,
                args=(training_id, "classification", model_size, pretrained, params),
                daemon=True
            )
            thread.start()
            self._training_threads[training_id] = thread

            return {
                "success": True,
                "training_id": training_id,
                "status": "started",
                "message": "Training started in background. Use yolo_get_training_status to check progress."
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start training: {str(e)}"
            }

    def get_training_status(self, training_id: str) -> dict[str, Any]:
        """Get status of async training.

        Args:
            training_id: Training task ID

        Returns:
            dict: Current status, progress, and metrics
        """
        if training_id not in self._training_status:
            return {
                "success": False,
                "error": "Training ID not found",
                "message": f"No training found with ID: {training_id}"
            }

        status_info = self._training_status[training_id].copy()

        # Calculate elapsed time
        if "start_time" in status_info:
            start_time = datetime.fromisoformat(status_info["start_time"])
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            status_info["elapsed_time_minutes"] = round(elapsed_minutes, 2)

            # Estimate remaining time
            if status_info["progress"] > 0 and status_info["status"] == "running":
                total_estimated = elapsed_minutes / (status_info["progress"] / 100)
                remaining = total_estimated - elapsed_minutes
                status_info["estimated_time_remaining_minutes"] = round(max(0, remaining), 2)

        # Clean up internal fields
        status_info.pop("params", None)
        status_info.pop("model_size", None)
        status_info.pop("pretrained", None)

        return {
            "success": True,
            "training_id": training_id,
            **status_info
        }

    # === Utility Methods ===

    def create_detection_yaml(
        self,
        train_dir: str,
        val_dir: str,
        class_names: list[str],
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create YAML config for detection training.

        Args:
            train_dir: Training images directory
            val_dir: Validation images directory
            class_names: List of class names
            output_path: Output path for YAML file (optional)

        Returns:
            dict: YAML file path and content
        """
        try:
            # Determine output path
            if output_path is None:
                output_path = str(self.output_dir / "data.yaml")

            # Create YAML structure
            yaml_content = {
                "path": str(Path(train_dir).parent.absolute()),
                "train": str(Path(train_dir).relative_to(Path(train_dir).parent)),
                "val": str(Path(val_dir).relative_to(Path(val_dir).parent)),
                "nc": len(class_names),
                "names": class_names
            }

            # Write YAML file
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

            return {
                "success": True,
                "yaml_path": str(output_path_obj.absolute()),
                "content": yaml_content
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create YAML config: {str(e)}"
            }

    def validate_model(
        self,
        model_path: str,
        data_yaml: str,
        imgsz: int = 640,
        batch: int = 16,
        **kwargs
    ) -> dict[str, Any]:
        """Validate trained model.

        Args:
            model_path: Path to model file (.pt)
            data_yaml: Path to data configuration file
            imgsz: Validation image size
            batch: Batch size, default 16
            **kwargs: Additional validation parameters

        Returns:
            dict: Validation metrics
        """
        try:
            # Load model
            model = YOLO(model_path)

            # Validate
            results = model.val(
                data=data_yaml,
                imgsz=imgsz,
                batch=batch,
                **kwargs
            )

            # Extract metrics
            metrics = {}
            if hasattr(results, 'box'):
                # Detection metrics
                metrics = {
                    "mAP50": float(results.box.map50),
                    "mAP50-95": float(results.box.map),
                    "precision": float(results.box.mp),
                    "recall": float(results.box.mr),
                }
            elif hasattr(results, 'top1'):
                # Classification metrics
                metrics = {
                    "top1_accuracy": float(results.top1),
                    "top5_accuracy": float(results.top5),
                }

            return {
                "success": True,
                "model_path": model_path,
                "metrics": metrics
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Validation failed: {str(e)}"
            }

    def export_model(
        self,
        model_path: str,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        """Export model to deployment format.

        Args:
            model_path: Path to model file (.pt)
            format: Export format (onnx/torchscript/openvino/engine), default "onnx"
            imgsz: Image size for export
            half: Use FP16 precision, default False
            **kwargs: Additional export parameters

        Returns:
            dict: Export result with file path
        """
        try:
            # Load model
            model = YOLO(model_path)

            # Export
            export_path = model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                **kwargs
            )

            return {
                "success": True,
                "model_path": model_path,
                "export_format": format,
                "export_path": str(export_path)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Export failed: {str(e)}"
            }

    # === Private Methods ===

    def _run_training_worker(
        self,
        training_id: str,
        task_type: str,
        model_size: str,
        pretrained: bool,
        params: dict
    ) -> None:
        """Background worker for async training.

        Args:
            training_id: Training task ID
            task_type: "detection" or "classification"
            model_size: Model size (n/s/m/l/x)
            pretrained: Use pretrained weights
            params: Training parameters
        """
        try:
            # Update status to running
            self._training_status[training_id]["status"] = "running"

            # Load model
            if task_type == "detection":
                model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
            else:
                model_name = f"yolov8{model_size}-cls.pt" if pretrained else f"yolov8{model_size}-cls.yaml"

            model = YOLO(model_name)

            # Add callbacks for progress tracking
            def on_epoch_end(trainer):
                epoch = trainer.epoch + 1
                total = params["epochs"]
                progress = int(epoch / total * 100)

                # Extract current metrics
                current_metrics = {}
                if task_type == "detection":
                    if hasattr(trainer, 'metrics') and hasattr(trainer.metrics, 'box'):
                        current_metrics = {
                            "mAP50": float(trainer.metrics.box.map50) if trainer.metrics.box.map50 else 0,
                        }
                elif task_type == "classification":
                    if hasattr(trainer, 'metrics'):
                        current_metrics = {
                            "top1_accuracy": float(trainer.metrics.top1) if hasattr(trainer.metrics, 'top1') else 0,
                        }

                # Update status
                self._training_status[training_id].update({
                    "current_epoch": epoch,
                    "progress": progress,
                    "current_metrics": current_metrics,
                })

            model.add_callback("on_train_epoch_end", on_epoch_end)

            # Train
            results = model.train(**params)

            # Format final results
            if task_type == "detection":
                final_results = self._format_detection_results(
                    results, training_id, params["project"], params["name"]
                )
            else:
                final_results = self._format_classification_results(
                    results, training_id, params["project"], params["name"]
                )

            # Update status to completed
            self._training_status[training_id].update({
                "status": "completed",
                "progress": 100,
                "results": final_results,
            })

        except Exception as e:
            # Update status to failed
            self._training_status[training_id].update({
                "status": "failed",
                "error": str(e),
            })

    def _format_detection_results(
        self,
        results,
        training_id: str,
        project: str,
        name: str
    ) -> dict[str, Any]:
        """Format detection training results.

        Args:
            results: Training results from YOLO
            training_id: Training task ID
            project: Project directory
            name: Experiment name

        Returns:
            dict: Formatted results
        """
        save_dir = Path(project) / name

        # Extract metrics
        metrics = {}
        if hasattr(results, 'box'):
            metrics = {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }

        # Find best model
        best_model = save_dir / "weights" / "best.pt"
        last_model = save_dir / "weights" / "last.pt"

        # Find result charts
        charts = []
        for chart_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
            chart_path = save_dir / chart_name
            if chart_path.exists():
                charts.append(str(chart_path))

        return {
            "success": True,
            "training_id": training_id,
            "model_path": str(best_model) if best_model.exists() else str(last_model),
            "results": {
                "metrics": metrics,
            },
            "logs_dir": str(save_dir),
            "charts": charts,
        }

    def _format_classification_results(
        self,
        results,
        training_id: str,
        project: str,
        name: str
    ) -> dict[str, Any]:
        """Format classification training results.

        Args:
            results: Training results from YOLO
            training_id: Training task ID
            project: Project directory
            name: Experiment name

        Returns:
            dict: Formatted results
        """
        save_dir = Path(project) / name

        # Extract metrics
        metrics = {}
        if hasattr(results, 'top1'):
            metrics = {
                "top1_accuracy": float(results.top1),
                "top5_accuracy": float(results.top5),
            }

        # Find best model
        best_model = save_dir / "weights" / "best.pt"
        last_model = save_dir / "weights" / "last.pt"

        # Find result charts
        charts = []
        for chart_name in ["results.png", "confusion_matrix.png"]:
            chart_path = save_dir / chart_name
            if chart_path.exists():
                charts.append(str(chart_path))

        return {
            "success": True,
            "training_id": training_id,
            "model_path": str(best_model) if best_model.exists() else str(last_model),
            "results": {
                "metrics": metrics,
            },
            "logs_dir": str(save_dir),
            "charts": charts,
        }


# Singleton pattern
_yolo_trainer: Optional[YOLOTrainer] = None


def get_yolo_trainer() -> YOLOTrainer:
    """Get or create global YOLO trainer instance."""
    global _yolo_trainer
    if _yolo_trainer is None:
        _yolo_trainer = YOLOTrainer()
    return _yolo_trainer
