from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from typing import List
from transformers import AutoModelForCausalLM
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Serve your reference model")
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="model_name_or_path")
    parser.add_argument("--max_parallel_requests", type=int, default=2, help="max_parallel_requests")
    parser.add_argument("--max_wait_time", type=float, default=0.1, help="max_wait_time")
    parser.add_argument("--port", type=int, default=8000, help="port")
    
    args = parser.parse_args()
    return args


executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(model_server.batch_processor())
    yield
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

class ModelInput(BaseModel):
    input_ids: List[List[int]]
    num_logits_to_keep: int


class ModelServer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=self.device, torch_dtype=torch.float16)
        self.model.eval()
        self.req_queue = asyncio.Queue()
        self.max_parallel_requests = args.max_parallel_requests
        self.max_wait_time = args.max_wait_time

    async def batch_processor(self):
        while True:
            request_list = []
            try:
                first_item = await self.req_queue.get()
                request_list.append(first_item)
                
                timeout = self.max_wait_time
                while len(request_list) < self.max_parallel_requests:
                    try:
                        item = await asyncio.wait_for(
                            self.req_queue.get(), 
                            timeout=timeout
                        )
                        request_list.append(item)
                    except asyncio.TimeoutError:
                        break
                        
                input_ids = torch.cat([item[0] for item in request_list])
                num_logits = max(item[1] for item in request_list)
                lengths = [len(item[0]) for item in request_list]
                
                outputs = await self.async_inference(input_ids, num_logits)
                torch.cuda.empty_cache()
                
                start_idx = 0
                for i, (_, _, future) in enumerate(request_list):
                    end_idx = start_idx + lengths[i]
                    future.set_result(outputs[start_idx:end_idx])
                    start_idx = end_idx
                    
            except Exception as e:
                for _, _, future in request_list:
                    future.set_exception(e)
                    
    @torch.no_grad()
    def get_per_token_logps(self, input_ids, num_logits_to_keep):
        logits = self.model(input_ids=input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        per_token_logps = []
        for logits_row, ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    async def async_inference(self, input_ids, num_logits_to_keep):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            functools.partial(self.get_per_token_logps, input_ids, num_logits_to_keep)
        )


@app.post("/predict")
async def predict(data: ModelInput):
    try:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        input_tensor = torch.tensor(data.input_ids).to(model_server.device)
        await model_server.req_queue.put((input_tensor, data.num_logits_to_keep, future))
        
        result = await future
        return {"log_probs": result.cpu().numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_device": str(model_server.device)}


if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    # args.model_name_or_path = "lm_models/Qwen2.5-0.5B-Instruct"
    # args.max_parallel_requests = 2
    # args.max_wait_time = 0.1
    model_server = ModelServer(args)
    uvicorn.run(app, port=args.port)