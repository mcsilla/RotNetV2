#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__init__ module for configs. Register your config file here by adding it's
entry in the CONFIG_MAP as shown.
"""

import config.rotnet90_tpu

CONFIG_MAP = {
    'rotnet90_tpu': config.rotnet90_tpu.CONFIG,
}
