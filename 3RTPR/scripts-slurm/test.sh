noisy_rate=0.5  #0.0 0.2 0.5 0.8
for p in /home/k64t/TPS/saves_a100giang/saves/CUHK-PEDES/20250718_101358_CUHK128-3456-nr-0.5_8479 \
 /home/k64t/TPS/saves_a100giang/saves/CUHK-PEDES/20250719_075152_CUHK128-3456-nr-0.5_3566  \
 /home/k64t/TPS/saves_a100giang/saves/CUHK-PEDES/20250721_034358_CUHK128-12345-nr-0.5_8369  \
 /home/k64t/TPS/saves_a100giang/saves/CUHK-PEDES/20250720_163152_CUHK64-12345-nr-0.5_6758  ;do
for DATASET_NAME in CUHK-PEDES; do
noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
python run.py  --cfg cfg-t4.yml \
  --d-names $DATASET_NAME --output_dir $p \
  --bs 16 --erpt 0.2 --test \
  --ldynamic  --ldynamic-t 2 --lossweight-sdm 0 --lossweight-trl 1  --sratio 0.4 --fusedim 5120  \
  --noisy_file $noisy_file  --noisy_rate $noisy_rate --lrx 0.1
done
done
