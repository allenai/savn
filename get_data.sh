#!/usr/bin/env bash


beaker dataset fetch --output=thor_glove ds_piguyn8nguzy
beaker dataset fetch --output=thor_offline_data ds_eawz8kfvv7mp

mkdir data

mv thor_glove data
mv thor_offline_data data