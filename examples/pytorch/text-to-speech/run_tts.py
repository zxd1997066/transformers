
import argparse
import os
import time

import torch
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN


MODEL_CLASSES = {
    "speechbrain/tts-tacotron2-ljspeech": (
        Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts"),
        HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
    )
}

def test(args, tacotron2, hifi_gan):
    total_sample = 0
    total_time = 0.0
    if args.compile:
        if args.backend == "cudagraphs":
            tacotron2.encode_text = torch.compile(tacotron2.encode_text, backend=args.backend)
        # elif args.backend == "zentorch":
        #     import zentorch
        #     import torch
        #     tacotron2.encode_text = torch.compile(tacotron2.encode_text, backend=args.backend)
        else:
            tacotron2.encode_text = torch.compile(tacotron2.encode_text, backend=args.backend, options={"freezing": True})
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter):
                elapsed = time.time()
                # Running the TTS
                mel_output, mel_length, alignment = tacotron2.encode_text(args.prompt)
                # Running Vocoder (spectrogram-to-waveform)
                waveforms = hifi_gan.decode_batch(mel_output)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                p.step()
                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_time += elapsed
                    total_sample += 1
    else:
        for i in range(args.num_iter):
            elapsed = time.time()
            # Running the TTS
            mel_output, mel_length, alignment = tacotron2.encode_text(args.prompt)
            # Running Vocoder (spectrogram-to-waveform)
            waveforms = hifi_gan.decode_batch(mel_output)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_time += elapsed
                total_sample += 1

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {:.3f} ms".format(latency))
    print("inference Throughput: {:.2f} samples/s".format(throughput))

    # Save the waverform
    if args.save_audio:
        torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'beit-large-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument("--model_name_or_path", type=str, default='speechbrain/tts-tacotron2-ljspeech', help="model name")
    parser.add_argument('--prompt', default="Mary had a little lamb", type=str, help='input text')
    parser.add_argument('--device', default="cpu", type=str, help='cpu, cuda or xpu')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=1, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=0, type=int, help='test warmup')
    parser.add_argument('--save_audio', action='store_true', default=False, help='save_audio')
    #
    parser.add_argument('--per_device_eval_batch_size', default=1, type=int, help='useless')
    parser.add_argument('--do_eval', action='store_true', default=False, help='useless')
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False, help='useless')
    parser.add_argument('--output_dir', default='', type=str, help='useless')
    parser.add_argument("--device_oob", type=str, default="cpu")
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
    args = parser.parse_args()
    print(args)

    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    # tacotron2 = Tacotron2.from_hparams(source=args.model_name_or_path, savedir="tmpdir_tts")
    # hifi_gan = HIFIGAN.from_hparams(source=args.model_name_or_path, savedir="tmpdir_vocoder")
    tacotron2, hifi_gan = MODEL_CLASSES[args.model_name_or_path]
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    if args.channels_last:
        tacotron2 = tacotron2.to(memory_format=torch.channels_last)
        hifi_gan = hifi_gan.to(memory_format=torch.channels_last)
        print("---- Use CL model")

    # start test
    if args.precision == "bfloat16":
        print("---- Use AMP bfloat16")
        with torch.autocast(enabled=True, device_type=args.device_oob, dtype=torch.bfloat16):
            test(args, tacotron2, hifi_gan)
    elif args.precision == "float16":
        print("---- Use AMP float16")
        with torch.autocast(enabled=True, device_type=args.device_oob, dtype=torch.half):
            test(args, tacotron2, hifi_gan)
    else:
        test(args, tacotron2, hifi_gan)
