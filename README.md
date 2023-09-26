This repo uses [Rerun](https://github.com/rerun-io/rerun)] to visualize [Shap-E](https://github.com/openai/shap-e) and [Point-E](https://github.com/openai/point-e) from OpenAI

https://github.com/rerun-io/point-shap-e/assets/2624717/507e9f3f-4951-4bde-a2f6-4d7073016bc6


# Installation
run the following command

```pip install -r requirements.txt```

# Running Examples
## Shap-E
```bash
python main_shap_e.py --prompt {YOUR_PROMPT} --log-diffusion
```
## Point-E
```bash
python main_point_e.py --prompt {YOUR_PROMPT} --log-diffusion
```

## Comparison
```bash
python comparison.py --prompt {YOUR_PROMPT}
```
