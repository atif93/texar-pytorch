#!/usr/bin/env bash
DATA_DIR="${DATA_DIR:-glue_data/}"
mkdir -p $DATA_DIR
cd $DATA_DIR
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

tasks="$1"
shift
for ARG in $@; do
    tasks="$tasks,$ARG"
done

if test -z "$tasks"
then
    python download_glue_data.py --data_dir .
else
    python download_glue_data.py --data_dir . --tasks "$tasks"
fi
rm -f download_glue_data.py*