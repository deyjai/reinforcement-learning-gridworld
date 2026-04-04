**Warehouse Robot Q-Learning Experiments**



This project is a comparative study of exploration strategies in Q-Learning.



**Setup**



Install the required Python packages:



```bash

pip install -r requirements.txt

```



**Running the Project**



From the root folder, run:



```bash

python -m src.main

```



This will execute the experiments, generate plots, and launch Pygame visualizations of the learned policies.



**Folder Structure**



├── src/

│   ├── main.py                  # Main script to run experiments

│   ├── algorithms/

│   │   ├── q\_learning\_epsilon.py

│   │   └── q\_learning\_bonus.py

│   ├── environment/

│   │   └── warehouse\_gridworld\_domain\_random.py

│   └── visualization.py         # Functions for plotting and Pygame policy visualization

├── results/

│   ├── plots/                   # Generated plots

│   └── logs/                    # Raw reward and success logs

├── requirements.txt

└── README.md



