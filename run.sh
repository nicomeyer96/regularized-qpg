# Note: Executing this script to re-compute the full data set requires approximately 950 processor-hours (~40 days) of compute
#       -- assuming a AMD Ryzen 9 5900X 12-Core CPU with a clock-speed of 3.7 GHz.
#       Consider instead downloading pre-computed data as described in README.

THREADS=0  # use all available cores

######################################################
#  Training with different levels of regularization  #
######################################################  ~180 processor-hours
for REG in 0.00 0.10 0.20 0.30 0.40 0.50
do
  for SEED in {0..99}
  do
    python train.py --seed "$SEED" --reg $REG --threads $THREADS
  done
done

##########################################
#  Testing robustness of trained models  #
##########################################  ~20 processor-hours
for REG in 0.00 0.10 0.20 0.30 0.40 0.50
do
  for SEED in {0..99}
  do
    python test.py --model reg="$REG"_"$SEED" --runs 100 --perturbate --threads $THREADS
  done
done

###################################################
#  Testing (2d) generalization of trained models  #
###################################################  ~170 processor-hours
for REG in 0.00 0.10 0.20 0.40
do
  for SEED in {0..99}
  do
    python test.py --model reg="$REG"_"$SEED" --runs 100 --angle --velocity --threads $THREADS
  done
done

#################################################################
#  Curriculum training with different levels of regularization  #
#################################################################  ~530 processor-hours
for REG in 0.00 0.10 0.20
do
  for SEED in {0..99}
  do
    python train_curriculum.py --seed "$SEED" --reg $REG --val 1 --episodes 10000 --threads $THREADS
  done
done

##############################################################
#  Testing (1d) generalization of curriculum-trained models  #
##############################################################  ~50 processor-hours
# (automatically skips the models that have not been trained to the required quality)
for REG in 0.00 0.10 0.20
do
  for SEED in {0..99}
  do
    for QUALITY in "[-0.25,0.25]" "[-0.75,0.75]" "[-1.25,1.25]" "[-1.75,1.75]"
    do
      python test.py --model curr"$QUALITY"_reg="$REG"_"$SEED" --runs 100 --velocity --threads $THREADS
    done
  done
done
