# vLLM-XPU OPs/Kernels breakdown
The OPs in a vLLM-XPU running might dispatch to vllm-xpu-kernels and native torch-xpu-ops in eager mode, and also intel-xpu-backend-for-triton in torch.compile mode. The project targets to visualize the dispathing for further analysising of the most important OPs to optimize.

## Visualization details
- Show the breakdown in a web page.
- There will be an box to search or type in the model ID like that on the huggingface site.
- A table to show the OPs with shape, dtype, call count, memory OP count, arithmetic intensity, etc. Note that the dynamic dims like sequence length and batch size could be shown in the table as variables.
- Rows for duplicated layers could be merged, and add a colum to indicated the repeatation count.
- Enable the selection between eager mode and torch.compile mode.

## steps
- Onece the model is selected, load the `config.json` of the model to show the summary of the model like arch, number of layers, dense or MoE, dtype, etc.
- Press the profile button to run and profile, then collect and show the OP dispatch breakdown.