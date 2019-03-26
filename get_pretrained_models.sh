#!/usr/bin/env bash


beaker dataset fetch --output=savn_pretrained.dat ds_ckg7l7sjpc93
beaker dataset fetch --output=nonadaptivea3c_pretrained.dat ds_5qwqizb24cd0

mkdir pretrained_models

mv savn_pretrained.dat pretrained_models
mv nonadaptivea3c_pretrained.dat pretrained_models