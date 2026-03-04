# NodeAgent hardware metrics (scraped from exporter, prefixed with miles_ft_node_)
NODE_GPU_AVAILABLE = "miles_ft_node_gpu_available"
NODE_XID_CODE_RECENT = "miles_ft_node_xid_code_recent"
NODE_DISK_AVAILABLE_BYTES = "miles_ft_node_disk_available_bytes"
NODE_NIC_UP = "miles_ft_node_nic_up"
NODE_GPU_TEMPERATURE = "miles_ft_node_gpu_temperature_celsius"

# Controller synthetic metrics (injected directly, no prefix)
TRAINING_JOB_STATUS = "training_job_status"

# MegatronAgent heartbeat metrics (scraped from exporter, no NodeAgent prefix)
TRAINING_ITERATION = "training_iteration"
TRAINING_PHASE = "training_phase"
