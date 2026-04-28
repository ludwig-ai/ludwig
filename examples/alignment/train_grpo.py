"""GRPO alignment training with Ludwig.

Group Relative Policy Optimization (Shao et al., 2024 — DeepSeek-R1) trains a language
model using a programmatic reward signal rather than preference pairs. Instead of
"chosen vs rejected" data, you supply a reward function that scores each generated
response. Ludwig samples grpo_num_generations completions per prompt, normalises
rewards within the group, and applies a clipped PPO-style update.

Usage (standalone):
    python train_grpo.py
    python train_grpo.py --epochs 2 --lr 3e-7 --output_dir my_run

Usage (CLI with pre-scored dataset):
    ludwig train --config config_grpo.yaml --dataset grpo_train.csv

Prerequisites:
    # Colab: !pip install "ludwig[llm]"
    pip install "ludwig[llm]"
    export HUGGING_FACE_HUB_TOKEN="<your_token>"
    # You also need access approval for meta-llama/Llama-3.1-8B on HuggingFace Hub.

How reward functions work with the Ludwig GRPO trainer
-------------------------------------------------------
The GRPO trainer expects a dataset where each row has:
  - prompt   : the input question
  - response : the correct / reference answer (used as a training target)

The reward function is applied *before* training, in a data preparation step that
scores each (prompt, response) pair and stores a "reward" column. During the GRPO
update the trainer uses those scores — together with the group-normalised advantages
computed across grpo_num_generations rollouts — to weight the policy gradient.

NOTE: If a future Ludwig version adds a reward_fn parameter directly on LudwigModel
or the GRPO trainer config, you can pass the callable there instead of pre-scoring.
Check the Ludwig changelog for that API addition.
"""

import argparse
import logging
import os
import re

import pandas as pd
import yaml

from ludwig.api import LudwigModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset — 100 math word problems generated inline; no download needed
# ---------------------------------------------------------------------------

MATH_PROBLEMS = [
    {
        "prompt": "A baker has 48 cookies. She puts them equally into 6 bags. How many cookies are in each bag?",
        "answer": "8",
    },
    {"prompt": "Tom has 15 apples. He gives 7 to his friend. How many apples does Tom have now?", "answer": "8"},
    {"prompt": "A train travels 60 miles per hour. How far does it travel in 3 hours?", "answer": "180"},
    {"prompt": "There are 24 students in a class. They form groups of 4. How many groups are there?", "answer": "6"},
    {"prompt": "Maria saves $12 each week. How much does she save in 5 weeks?", "answer": "60"},
    {"prompt": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?", "answer": "40"},
    {
        "prompt": "Jake has 3 boxes of crayons. Each box has 16 crayons. How many crayons does he have in total?",
        "answer": "48",
    },
    {
        "prompt": "A shop sells 35 items on Monday and 47 items on Tuesday. How many items were sold in total?",
        "answer": "82",
    },
    {
        "prompt": "There are 100 balloons. 38 are red and the rest are blue. How many blue balloons are there?",
        "answer": "62",
    },
    {"prompt": "A car travels 90 km in 2 hours. What is its average speed in km/h?", "answer": "45"},
    {"prompt": "Lucy reads 20 pages per day. How many pages does she read in 7 days?", "answer": "140"},
    {"prompt": "A box holds 12 eggs. How many eggs are in 9 boxes?", "answer": "108"},
    {"prompt": "A pool has 500 litres of water. 175 litres evaporate. How many litres remain?", "answer": "325"},
    {"prompt": "There are 7 shelves with 9 books on each shelf. How many books are there in total?", "answer": "63"},
    {"prompt": "Sam earns $8 per hour. How much does he earn in 6 hours?", "answer": "48"},
    {"prompt": "A garden is 12 m long and 7 m wide. What is its perimeter?", "answer": "38"},
    {"prompt": "There are 5 rows of chairs with 14 chairs in each row. How many chairs are there?", "answer": "70"},
    {
        "prompt": "Anna bakes 4 trays of muffins. Each tray has 12 muffins. How many muffins does she bake?",
        "answer": "48",
    },
    {"prompt": "A rope is 72 cm long. It is cut into 8 equal pieces. How long is each piece?", "answer": "9"},
    {"prompt": "Ben has 45 stickers. He gives 18 to his sister. How many stickers does Ben have left?", "answer": "27"},
    {
        "prompt": "A cinema sold 256 tickets on Saturday and 198 on Sunday. "
        "How many tickets were sold over the weekend?",
        "answer": "454",
    },
    {"prompt": "There are 360 minutes in 6 hours. How many minutes are in 1 hour?", "answer": "60"},
    {"prompt": "A cyclist rides 15 km each day. How far does she ride in 4 days?", "answer": "60"},
    {"prompt": "A bookstore has 5 shelves with 30 books each. How many books does it have in total?", "answer": "150"},
    {"prompt": "A pizza is cut into 8 slices. If 3 slices are eaten, how many slices remain?", "answer": "5"},
    {"prompt": "There are 4 quarters in a dollar. How many quarters are in $7?", "answer": "28"},
    {"prompt": "A farmer has 120 eggs. He puts them in cartons of 12. How many cartons does he fill?", "answer": "10"},
    {
        "prompt": "A school has 480 pupils split equally across 6 classes. How many pupils are in each class?",
        "answer": "80",
    },
    {
        "prompt": "An ice cream shop sold 45 cones on Friday and 67 on Saturday. How many cones were sold in total?",
        "answer": "112",
    },
    {"prompt": "A factory makes 250 units per day. How many units does it make in 5 days?", "answer": "1250"},
    {
        "prompt": "There are 18 players in a tournament. They are split into teams of 3. How many teams are there?",
        "answer": "6",
    },
    {
        "prompt": "A jar holds 96 sweets. If 4 children share them equally, how many sweets does each child get?",
        "answer": "24",
    },
    {"prompt": "A plane flies 800 km in 2 hours. What is its average speed?", "answer": "400"},
    {
        "prompt": "A garden has 5 rows of flowers with 11 flowers in each row. How many flowers are there?",
        "answer": "55",
    },
    {"prompt": "James has $200. He spends $74 on shoes. How much money does he have left?", "answer": "126"},
    {"prompt": "A recipe uses 3 cups of flour per cake. How many cups are needed for 7 cakes?", "answer": "21"},
    {"prompt": "There are 50 chairs in a hall. 13 are occupied. How many chairs are empty?", "answer": "37"},
    {"prompt": "A clock ticks 60 times per minute. How many times does it tick in 5 minutes?", "answer": "300"},
    {
        "prompt": "A runner completes a 400 m lap in 80 seconds. How many laps does she run in 400 seconds?",
        "answer": "5",
    },
    {"prompt": "There are 3 packs of pens with 12 pens each. How many pens are there in total?", "answer": "36"},
    {"prompt": "A tank holds 200 gallons. It is currently 40% full. How many gallons are in the tank?", "answer": "80"},
    {
        "prompt": "A store has 84 items. They are arranged in 7 equal rows. How many items are in each row?",
        "answer": "12",
    },
    {
        "prompt": "There are 9 months until the concert. How many weeks is that (assuming 4 weeks per month)?",
        "answer": "36",
    },
    {"prompt": "A frog jumps 3 m each jump. How far does it jump in 15 jumps?", "answer": "45"},
    {"prompt": "A box of pencils costs $3. How much do 11 boxes cost?", "answer": "33"},
    {"prompt": "There are 144 hours in 6 days. How many hours are in 1 day?", "answer": "24"},
    {"prompt": "A pond has 300 fish. 75 are caught and released. How many fish remain?", "answer": "225"},
    {"prompt": "A school bus seats 40 students. How many students can travel in 3 buses?", "answer": "120"},
    {
        "prompt": "There are 11 teams in a league. Each team plays every other team once. How many matches are there?",
        "answer": "55",
    },
    {"prompt": "A bakery makes 60 loaves a day. How many loaves does it make in 2 weeks?", "answer": "840"},
    {"prompt": "A square has sides of 9 cm. What is its perimeter?", "answer": "36"},
    {"prompt": "An author writes 500 words per hour. How many words does she write in 3 hours?", "answer": "1500"},
    {"prompt": "There are 72 hours in 3 days. How many hours are in 5 days?", "answer": "120"},
    {"prompt": "A store discounts a $50 item by 20%. What is the sale price?", "answer": "40"},
    {"prompt": "A farmer plants 8 seeds in each row. He has 9 rows. How many seeds does he plant?", "answer": "72"},
    {"prompt": "A car park has 6 levels with 45 spaces each. How many spaces are there in total?", "answer": "270"},
    {"prompt": "A marathon is 42 km. A runner has covered 28 km. How many km remain?", "answer": "14"},
    {"prompt": "There are 30 days in a month. How many days are in 4 months?", "answer": "120"},
    {"prompt": "A vending machine sells 15 drinks per hour. How many drinks does it sell in 8 hours?", "answer": "120"},
    {"prompt": "There are 6 strings on a guitar. How many strings are on 9 guitars?", "answer": "54"},
    {"prompt": "A worker earns $15 per hour and works 8 hours. How much does she earn?", "answer": "120"},
    {"prompt": "A square room has an area of 64 m². What is the length of each side?", "answer": "8"},
    {"prompt": "A jar has 5 red marbles and 8 blue marbles. How many marbles are there in total?", "answer": "13"},
    {"prompt": "A truck carries 2 tonnes per trip. How many tonnes does it carry in 7 trips?", "answer": "14"},
    {"prompt": "There are 32 students. Half are girls. How many girls are there?", "answer": "16"},
    {"prompt": "A patio is 6 m wide and 9 m long. What is its area?", "answer": "54"},
    {"prompt": "A swimmer does 50 laps per session. How many laps does she do in 6 sessions?", "answer": "300"},
    {"prompt": "A bag of rice weighs 5 kg. How much do 8 bags weigh?", "answer": "40"},
    {"prompt": "There are 7 days in a week. How many days are in 13 weeks?", "answer": "91"},
    {"prompt": "A factory produces 1200 items in 4 hours. How many items per hour?", "answer": "300"},
    {
        "prompt": "A road is 3.5 km long. Two cars start at each end and drive toward each other "
        "at 1.75 km/h each. How long until they meet (in hours)?",
        "answer": "1",
    },
    {"prompt": "There are 18 biscuits. Each person eats 3. How many people can eat?", "answer": "6"},
    {
        "prompt": "A pool requires 8 hours to fill. After 5 hours, how much is left to fill (as a fraction)?",
        "answer": "3/8",
    },
    {"prompt": "A box has 4 layers with 25 chocolates per layer. How many chocolates are in the box?", "answer": "100"},
    {"prompt": "A phone battery lasts 12 hours. After 9 hours of use, what percentage remains?", "answer": "25"},
    {"prompt": "There are 40 red and 25 blue tiles. How many tiles are there in total?", "answer": "65"},
    {"prompt": "A rope is 54 m long. It is divided into 9 equal pieces. How long is each piece?", "answer": "6"},
    {"prompt": "A bag has 3 green, 4 yellow, and 5 purple balls. How many balls are there?", "answer": "12"},
    {"prompt": "Each page has 30 lines. How many lines are on 8 pages?", "answer": "240"},
    {"prompt": "A worker packs 6 boxes per hour. How many boxes in 9 hours?", "answer": "54"},
    {"prompt": "There are 1000 metres in a kilometre. How many metres in 7.5 km?", "answer": "7500"},
    {"prompt": "A container holds 5 litres. How many containers are needed for 35 litres?", "answer": "7"},
    {
        "prompt": "A car travels 110 km in 2 hours. How far does it travel in 5 hours at the same speed?",
        "answer": "275",
    },
    {"prompt": "There are 52 cards in a deck. How many cards are in 3 decks?", "answer": "156"},
    {"prompt": "A hotel has 12 floors with 18 rooms each. How many rooms are there?", "answer": "216"},
    {
        "prompt": "A pie is split into 6 equal slices. Two people each eat 2 slices. How many slices remain?",
        "answer": "2",
    },
    {"prompt": "A printing press produces 40 pages per minute. How many pages in 15 minutes?", "answer": "600"},
    {"prompt": "There are 200 pupils. 45% are boys. How many boys are there?", "answer": "90"},
    {"prompt": "A wall is 3 m tall and 8 m wide. What is its area?", "answer": "24"},
    {"prompt": "A car uses 6 litres of fuel per 100 km. How much fuel is needed for 300 km?", "answer": "18"},
    {"prompt": "A team scores 3 goals in each of 6 matches. How many goals in total?", "answer": "18"},
    {"prompt": "There are 11 rows of seats with 20 seats each. How many seats are there?", "answer": "220"},
    {"prompt": "A box contains 50 nails. After using 23, how many remain?", "answer": "27"},
    {"prompt": "A wheel turns 360 degrees in one full rotation. How many degrees in 5 rotations?", "answer": "1800"},
    {"prompt": "A shop receives 3 deliveries of 40 items each. How many items did it receive?", "answer": "120"},
    {"prompt": "There are 15 biscuits on a plate. 6 are eaten. How many remain?", "answer": "9"},
]


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def reward_fn(prompt: str, response: str) -> float:
    """Score a model response by checking if it contains the correct numerical answer.

    Returns 1.0 if the response contains the expected answer, 0.0 otherwise. The correct answer is looked up from the
    global ANSWER_LOOKUP dict which is built from MATH_PROBLEMS at startup.

    This is a simple exact-match reward. Real applications might use an LLM judge, a code execution sandbox, or a
    symbolic verifier.
    """
    expected = ANSWER_LOOKUP.get(prompt.strip())
    if expected is None:
        return 0.0
    # Accept the answer if it appears as a standalone number/fraction in the response
    pattern = r"(?<![.\d])" + re.escape(expected) + r"(?![.\d])"
    return 1.0 if re.search(pattern, response) else 0.0


# Map from prompt text → expected answer string, built once at module load time.
ANSWER_LOOKUP: dict[str, str] = {p["prompt"]: p["answer"] for p in MATH_PROBLEMS}


# ---------------------------------------------------------------------------
# Dataset preparation — apply reward_fn to produce a scored DataFrame
# ---------------------------------------------------------------------------


def build_dataset(reward_function) -> pd.DataFrame:
    """Build a training DataFrame by applying the reward function to each example.

    For GRPO the dataset needs at minimum:
      - prompt   : the input text
      - response : the reference / target text
      - reward   : a float score for each (prompt, response) pair

    NOTE: The Ludwig GRPO trainer does not yet accept a reward_fn callable at
    train-time (as of v0.11.dev). Instead, rewards are pre-computed here and
    stored in a 'reward' column that the trainer reads from the dataset. If a
    future version exposes a reward_fn parameter on LudwigModel or the GRPO
    config, you can pass `reward_fn=reward_function` there directly.
    """
    rows = []
    for item in MATH_PROBLEMS:
        prompt = item["prompt"]
        response = item["answer"]
        reward = reward_function(prompt, response)
        rows.append({"prompt": prompt, "response": response, "reward": reward})
    df = pd.DataFrame(rows)
    logger.info(
        "Dataset built: %d examples, mean reward=%.3f",
        len(df),
        df["reward"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(epochs: int, learning_rate: float, batch_size: int) -> dict:
    raw = f"""
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapter:
  type: lora
  r: 16
  alpha: 32

trainer:
  type: grpo
  epochs: {epochs}
  learning_rate: {learning_rate}
  batch_size: {batch_size}
  gradient_accumulation_steps: 16
  grpo_beta: 0.04
  grpo_epsilon: 0.2
  grpo_num_generations: 4

input_features:
  - name: prompt
    type: text

output_features:
  - name: response
    type: text

backend:
  type: local
"""
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------


def check_gpu():
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("GPU detected: %s (%.1f GiB VRAM)", name, vram)
            if vram < 20:
                logger.warning(
                    "Only %.1f GiB VRAM detected. Llama-3.1-8B requires at least 40 GiB "
                    "for GRPO training. Consider using a smaller base model or enabling "
                    "quantisation (e.g. bitsandbytes 4-bit).",
                    vram,
                )
        else:
            logger.warning(
                "No GPU detected. GRPO training on a 7-8B model will be extremely slow on CPU. "
                "On Colab, go to Runtime > Change runtime type and select a GPU."
            )
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GRPO alignment training with Ludwig.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--experiment_name", default="math_grpo")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    # --- HuggingFace token ---
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise OSError(
            "Set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) before running. "
            "You also need access approval for meta-llama/Llama-3.1-8B on HuggingFace Hub."
        )

    check_gpu()

    # --- Build dataset with pre-computed rewards ---
    df = build_dataset(reward_fn)

    # --- Build Ludwig config ---
    config = build_config(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    # --- Train ---
    model = LudwigModel(config=config, logging_level=logging.INFO)

    train_stats, _, output_directory = model.train(
        dataset=df,
        experiment_name=args.experiment_name,
        output_directory=args.output_dir,
    )

    print(f"\nTraining complete. Results saved to: {output_directory}")
    print("To upload the aligned model to HuggingFace Hub:")
    print(f"    ludwig upload hf_hub -r <your_org>/<model_name> -m {output_directory}")


if __name__ == "__main__":
    main()
