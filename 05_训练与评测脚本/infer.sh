#!/bin/bash
set -e

bash "$(dirname "$0")/推理脚本/infer_chat_examples.sh" "${1:-xiaolaoshi}"
