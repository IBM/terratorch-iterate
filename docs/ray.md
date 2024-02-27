# Parallelizing with ray

To parallelize with ray, some additional steps are necessary.

Instructions for CCC follow.

!!! warning
    Running anything on a ray cluster is bound to bring about more cryptic messages and harder debugging.
    Please make sure your benchmark works for the single node case first,

## Set up a ray cluster

If on CCC, start a head node with:

```sh
export RAY_PORT=20022 # or any other port you like

jbsub -queue x86_24h -cores 2 -mem 32g ray start --head --port $RAY_PORT --dashboard-port $((RAY_PORT + 1)) --include-dashboard True --dashboard-host 0.0.0.0 --object-store-memory 10000000000 --num-cpus 0 --num-gpus 0 --temp-dir /tmp
```

Find out the address of your ray head with `bpeek <ccc process number>`
This will also tell you the url where you can check the ray cluster.

Then, launch your workers:

```sh
jbsub -queue <ccc_queue> -cores <nodes x cpu + gpu> -mem <mem> ./start_ray_workers.sh -a <ray_head_ip>:$RAY_PORT
```

You may have to run `chmod +x start_ray_workers.sh`.
You can provide multiple nodes in the command above. Each GPU will be used to launch a task. For each GPU, make sure there are at least 6 CPUs cores.

## Run the script

You can now run `ray_benchmark <ray_head_ip_with_no_port> --config <your_config>`.

## :::benchmark.benchmark_ray.benchmark_backbone