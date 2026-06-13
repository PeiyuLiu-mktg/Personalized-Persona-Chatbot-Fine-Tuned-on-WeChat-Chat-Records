#!/bin/bash
set -e

bash "$(dirname "$0")/训练脚本/train_${1:-xiaolaoshi}_history.sh"
