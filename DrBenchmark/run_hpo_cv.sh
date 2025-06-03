#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=0 nohup /bin/bash run_hpo_cv.sh transbert-bio-fr &
stage=2
nfolds=5
model_name=$1

# Download datasets
if [ $stage -le 0 ]
then
    python download_datasets_locally.py
fi


if [ $stage -le 1 ]
then
  echo "Downloading $model_name"
  model_name_dir=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | tr '/' '_')
  huggingface-cli download $model_name --local-dir ./models/$model_name_dir
  echo "Saving model in ./models/$model_name_dir"
fi

#Corpus CAS
if [ $stage -le 2 ]
then
  pushd recipes/cas/scripts/
 #POS
 for fold in `seq 1 1 $nfolds`
 do
   echo Corpus CAS Task
   echo Starting fold ${fold}
   ./run_task_1_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

#Corpus CLISTER
if [ $stage -le 3 ]
then
 pushd recipes/clister/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus CLISTER Task
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

# Corpus DEFT2020
if [ $stage -le 12 ]
then
 pushd recipes/deft2020/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus DEFT2020 Task 1
     echo Starting fold ${fold}
     ./run_task_1_hpo_cv.sh ${model_name} ${fold}
 done

 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus DEFT2020 Task 2
     echo Starting fold ${fold}
     ./run_task_2_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

#CORPUS Diamed
if [ $stage -le 4 ]
then
 pushd recipes/diamed/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus Diamed Task
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

#Corpus E3C
# French_clinical
if [ $stage -le 5 ]
then
 pushd recipes/e3c/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus E3C Task French_clinical
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} French_clinical ${fold}
 done

 #  French_temporal
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus E3C Task French_temporal
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} French_temporal ${fold}
 done
 popd
fi

#Corpus ESSAI
if [ $stage -le 6 ]
then
  pushd recipes/essai/scripts/
  for fold in `seq 1 1 $nfolds`
  do
      echo Corpus ESSAI
      echo Starting fold ${fold}
      ./run_task_1_hpo_cv.sh ${model_name} ${fold}
  done
  popd
fi

#Corpus FrenchMedMCQA
if [ $stage -le 7 ]
then
 pushd recipes/frenchmedmcqa/scripts/
 # CLS
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus FrenchMedMCQA CLS
     echo Starting fold ${fold}
     ./run_task_2_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

#Corpus MantraGSC
if [ $stage -le 8 ]
then
 pushd recipes/mantragsc/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus MantraGSC
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} all ${fold}
 done
 popd
fi

#Corpus Morfitt
if [ $stage -le 9 ]
then
 pushd recipes/morfitt/scripts/
 for fold in `seq 1 1 $nfolds`
 do
   echo Corpus Morfitt
   echo Starting fold ${fold}
   ./run_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi

#Corpus PXCorpus
# NER
if [ $stage -le 10 ]
then
 pushd recipes/pxcorpus/scripts/
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus PXCorpus NER
     echo Starting fold ${fold}
     ./run_task_1_hpo_cv.sh ${model_name} ${fold}
 done

 # CLS
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus PXCorpus CLS
     echo Starting fold ${fold}
     ./run_task_2_hpo_cv.sh ${model_name} ${fold}
 done
 popd
fi


#Corpus QUAERO
if [ $stage -le 11 ]
then
 pushd recipes/quaero/scripts/
 #EMEA
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus QUAERO EMEA
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} emea ${fold}
 done

 #MEDLINE
 for fold in `seq 1 1 $nfolds`
 do
     echo Corpus QUAERO MEDLINE
     echo Starting fold ${fold}
     ./run_hpo_cv.sh ${model_name} medline ${fold}
 done
 popd
fi
