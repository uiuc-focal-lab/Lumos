import carla
import time
import subprocess
import os
import threading
import json
from pathlib import Path

class SimpleWeatherExperiment:
    
    def __init__(self, carla_host='localhost', carla_port=2000):
        self.host = carla_host
        self.port = carla_port
        self.client = None
        self.world = None
        self.camera = None
        self.image_count = 0
        
    def connect_to_carla(self):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"Connected to CARLA at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False
    
    def wait_for_ego_vehicle(self, timeout=30):
        """Wait for ego vehicle to spawn in the world"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            vehicles = self.world.get_actors().filter('vehicle.*')
            if len(vehicles) > 0:
                # Find the vehicle with the ego role (or just take the first one)
                for vehicle in vehicles:
                    if hasattr(vehicle, 'attributes') and vehicle.attributes.get('role_name') == 'ego':
                        return vehicle
                    if hasattr(vehicle, 'type_id') and vehicle.type_id == "vehicle.lincoln.mkz_2017":
                        # If we find a vehicle but no ego role, return it
                        return vehicle
                # If no ego role, take the first vehicle
                return vehicles[0]
            time.sleep(0.1)
        
        return None
    
    def setup_camera(self, ego_vehicle, output_dir):
        """Setup camera on ego vehicle"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get camera blueprint
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find('sensor.camera.rgb')
            
            # Configure camera
            camera_bp.set_attribute('image_size_x', '1920')
            camera_bp.set_attribute('image_size_y', '1080')
            camera_bp.set_attribute('fov', '120')
            camera_bp.set_attribute('sensor_tick', '0.1')  # 10 FPS
            
            # Camera position (dashboard view)
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=1.4),  # Dashboard position
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            
            # Spawn camera
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
            
            # Setup image saving
            self.image_count = 0
            self.camera.listen(lambda image: self.save_image(image, output_dir))
            
            print(f"Camera attached, saving to: {output_dir}")
            return True
            
        except Exception as e:
            print(f"Failed to setup camera: {e}")
            return False
    
    def save_image(self, image, output_dir):
        """Save camera image"""
        try:
            self.image_count += 1
            filename = f"frame_{image.frame:06d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save_to_disk(filepath)
            
            # Print every 10th frame to avoid spam
            if self.image_count % 10 == 0:
                print(f"  Saved {self.image_count} images...")
                
                
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def cleanup_camera(self):
        """Clean up camera sensor"""
        if self.camera:
            try:
                self.camera.destroy()
                self.camera = None
                print("Camera cleaned up")
            except:
                pass
    
    def run_single_experiment(self, scenario_path, seed, output_dir, duration=30):
        """
        Run a single experiment: Scenic + Image Capture
        
        Args:
            scenario_path: Path to .scenic file
            seed: Random seed
            output_dir: Where to save images
            duration: How long to run (seconds)
        """
        
        print(f"\nStarting experiment: Seed {seed}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Connect to CARLA
        if not self.connect_to_carla():
            return False
        
        # Start the camera monitoring in a separate thread
        camera_thread = threading.Thread(
            target=self.monitor_and_capture, 
            args=(output_dir, duration + 10)  # Extra time buffer
        )
        camera_thread.daemon = True
        camera_thread.start()
        
        # Run Scenic simulation
        success = self.run_scenic_simulation(scenario_path, seed, duration)
        
        # Wait a bit for final images
        time.sleep(2)
        
        # Cleanup
        self.cleanup_camera()
        
        print(f"{'SUCCESS' if success else 'FAILED'}: {self.image_count} images captured")
        
        # Save metadata
        metadata = {
            'seed': seed,
            'scenario': str(scenario_path),
            'images_captured': self.image_count,
            'success': success
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return success
    
    def monitor_and_capture(self, output_dir, max_duration):
        """Monitor for ego vehicle and start capturing"""
        try:
            print("Waiting for ego vehicle...")
            
            # Wait for ego vehicle to appear
            ego_vehicle = self.wait_for_ego_vehicle()
            
            if ego_vehicle is None:
                print("No ego vehicle found")
                return
            
            print(f"Found ego vehicle: {ego_vehicle.type_id}")
            
            # Setup camera
            if not self.setup_camera(ego_vehicle, output_dir):
                return
            
            # Wait for the simulation to complete
            time.sleep(max_duration)
            
        except Exception as e:
            print(f"Monitor error: {e}")
    
    def run_scenic_simulation(self, scenario_path, seed, duration):
        """Run the Scenic simulation"""
        scenic_duration = max(duration, 650)
        try:
            # Prepare Scenic command
            cmd = [
                'scenic', str(scenario_path),
                '--simulate',
                '--2d',
                '--model', 'scenic.simulators.carla.model',
                '--param', 'render', '0',
                '--time', f'{scenic_duration}',
                '--seed', f'{seed}',
            ]
            
            print(f"Running Scenic: {' '.join(cmd)}")
            
            # Run with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=min(scenic_duration, 20))
            print(result.stdout)
            print(result.stderr)
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("Scenic simulation timed out")
            return True  # Still consider success if we got images
        except Exception as e:
            print(f"Scenic simulation error: {e}")
            return False


def run_weather_experiment(scenario_file, base_output_dir):
    
    # Configuration
    seeds = list(range(1, 2500, 10))
    DURATION = 5
    
    
    # Initialize experiment
    experiment = SimpleWeatherExperiment()
    
    # Run all combinations
    results = []
    total_runs = len(seeds)
    current_run = 0
    
    for seed in seeds:
        if experiment.client is not None:
            experiment.world = experiment.client.reload_world()
        current_run += 1
        print(f"\n{'='*60}")
        print(f"RUN {current_run}/{total_runs}: Seed {seed}")
        print(f"{'='*60}")
        
        # Output directory for this run
        run_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        if os.path.exists(run_output_dir) and len(os.listdir(run_output_dir)) > 1:
            print(f"Output directory {run_output_dir} already exists, skipping...")
            continue
        # Run experiment
        success = experiment.run_single_experiment(
            scenario_file, seed, run_output_dir, duration=DURATION
        )
        
        results.append({
            'seed': seed,
            'success': success,
            'output_dir': run_output_dir
        })
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/total_runs*100:.1f}%")
    
    print(f"\nResults saved to: {base_output_dir}/")
    print("Each successful run contains:")
    print("  - frame_XXXXXX.png files (captured images)")
    print("  - metadata.json (run information)")
    
    return results


if __name__ == "__main__":
    print("CARLA Weather Counterfactual Experiment")
    print("=" * 50)
    print("REQUIREMENTS:")
    print("1. CARLA server running on localhost:2000")
    print("2. Scenic installed and in PATH")
    print("3. Python carla package installed")
    print("\nPress Enter to start, or Ctrl+C to cancel...")
    
    for scenario_file in ['weather.scenic']:
        try:
            scenario_basename =scenario_file.split('.')[0]
            base_output_dir = f'{scenario_basename}_results'
            os.makedirs(base_output_dir, exist_ok=True)
            results = run_weather_experiment(scenario_file, base_output_dir)
        except Exception as e:
            print(f"\nExperiment failed: {e}")