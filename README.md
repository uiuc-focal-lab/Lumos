This directory contains the basic setup for running Scenic scenarios in CARLA to create and store images for scenes based on scene graphs.

- `run_scenic.py`: This script executes the scenic scenarios and captures data.
- `*.scenic`: These files define the specific scenarios (e.g., `obstacle_clear.scenic`, `obstacle_rain.scenic`, etc.).
- `certificate.py`: This script runs certification on the generated images using VQA models (Qwen or LLaVA).

## Setup

### Requirements
1.  **CARLA Simulator**: You need a running CARLA server.
    *   Download CARLA (e.g., 0.9.14 or newer).
    *   Run the server: `./CarlaUE4.sh` (defaults to localhost:2000).
2.  **Scenic**: Install the Scenic language.
    *   `pip install scenic`
3.  **Python Packages**:
    *   `carla`: Ensure the Python client matches your server version.
    *   `transformers`, `torch`, `Pillow`, `statsmodels` (for certification).

### Running Scenarios
1.  Start CARLA server.
2.  Run the experiment script:
    ```bash
    python run_scenic.py
    ```

### Running Certification
To certify the results using a VQA model:
```bash
python certificate.py --question "Is there a car?" --answer "yes" --image_dir "path/to/images" --model_name "qwen"
```

### Certifications from Paper

1. **Brake with an obstacle very close**
   - **Scenario Files**: `obstacle_*.scenic`
   - **Question**: Answer with only ‘Yes’ or ‘No’: Is it necessary for the vehicle to stop here?
   - **Correct Answer**: Yes

2. **Building-route blockage check**
   - **Scenario Files**: `obstacle_*.scenic`
   - **Description**: Ego->(1-3m ahead) Barrier/Vehicle
   - **Question**: Answer with only ‘Yes’ or ‘No’: Does a building block the vehicle’s route?
   - **Correct Answer**: No

3. **Right-turn feasibility**
   - **Scenario Files**: `right_*.scenic`
   - **Description**: Ego->(2m)->Intersection with right turn
   - **Question**: Answer with only ‘Yes’ or ‘No’: Is a right turn physically possible here?
   - **Correct Answer**: Yes
