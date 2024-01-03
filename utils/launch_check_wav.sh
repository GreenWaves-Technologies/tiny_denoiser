#!/usr/bin/env bash

# Traitement des options en ligne de commande
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --build-dir)
      GAPTEST_BUILD_DIR="$2"
      shift
      ;;
    --test-dir)
      GAPTEST_TEST_DIR="$2"
      shift
      ;;
    *)
      echo "Usage: $0 --build-dir <GAPTEST_BUILD_DIR> --test-dir <GAPTEST_TEST_DIR>"
      exit 1
      ;;
  esac
  shift
done

# Vérifier si les options obligatoires sont présentes
if [ -z "$GAPTEST_BUILD_DIR" ] || [ -z "$GAPTEST_TEST_DIR" ]; then
  echo "Usage: $0 --build-dir <GAPTEST_BUILD_DIR> --test-dir <GAPTEST_TEST_DIR>"
  exit 1
fi

# Vos commandes avec les variables
${GAPTEST_TEST_DIR}/utils/check_wav.py --input="${GAPTEST_BUILD_DIR}/output_p232_334.wav" --ref="${GAPTEST_TEST_DIR}/utils/golden.wav" --tolerance=1000 --skiped=15
