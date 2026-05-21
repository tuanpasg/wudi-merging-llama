
conda create -y -n wudi python=3.10.9
conda activate wudi 

cd /workspace/
git clone https://github.com/tuanpasg/wudi-merging-llama.git

cd /workspace/wudi-merging-llama/nlp_roberta
pip install -r requirements_llama.txt

cd /workspace/wudi-merging-llama/nlp_roberta/llama
python merge_llama.py \
 --out /workspace/outs/wudi \
 --merge_method wudi_merge \
 --scaling 1 \
 --wudi_device cuda \
 --exclude ".*embed.*" ".*lm_head.*"