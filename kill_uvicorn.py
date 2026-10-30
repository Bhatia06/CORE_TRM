import psutil
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = p.info['cmdline'] or []
        if 'uvicorn' in ' '.join(cmdline):
            print('Killing', p.info['pid'], cmdline)
            p.kill()
    except:
        pass