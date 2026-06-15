Kokoro Voice Directory
====================

This directory is part of the source distribution and does not contain
voice files. Voice files (.pt) are not bundled with the package.

Cache location
--------------

Voice files are resolved from the user cache directory:

    ~/.cache/champi-tts/voices/

Place your Kokoro voice files (.pt) in that directory before use.

CLI helpers
-----------

Print the cache directory path:

    champi-tts voices cache-dir

List voices already present in the cache:

    champi-tts voices list

Obtaining voice files
---------------------

Voice files are PyTorch tensors saved with torch.save(). Download them
from the Kokoro project or create your own and place them in the cache
directory shown above.

Example filenames:
- af_bella.pt
- af_sarah.pt
- am_adam.pt

The voice name (filename without .pt) is used to identify the voice
when calling synthesis commands.
