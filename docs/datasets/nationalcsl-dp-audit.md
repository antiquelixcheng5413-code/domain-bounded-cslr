# NationalCSL-DP access and label audit

Audit date: 2026-06-15.

Current status: temporary primary candidate for implementation planning and adapter work. It is
not the final locked experiment dataset until the remaining gates are complete.

## Source record

| Field | Verified value |
|---|---|
| Dataset | NationalCSL-DP |
| Article | <https://www.nature.com/articles/s41597-025-04986-x> |
| Data record | <https://figshare.com/articles/media/NationalCSL-DP/27261843> |
| DOI | `10.6084/m9.figshare.27261843.v3` |
| License | CC BY 4.0 |
| Vocabulary | 6,707 isolated Chinese National Sign Language signs |
| Samples | 134,140 videos |
| Signers | 10: eight deaf and two hearing signers proficient in CNSL |
| Views | Synchronized front and left |
| Published image representation | 256 x 256 extracted frames |
| Figshare package size | 18,642,323,021 bytes across 23 files |

The article states that the complete raw-video collection exceeds 1.8 TB. Figshare publishes
all extracted image frames and only a small raw-video sample. The project must therefore use
the image representation for the main experiment and use sample videos only to validate the
video pipeline.

## Integrity checks

The downloaded metadata files are kept outside the Git repository under the local cache
`D:\FYP_downloads\dataset-audit`.

| File | Size | Expected MD5 | Result |
|---|---:|---|---|
| `gloss.csv` | 172,465 bytes | `b62074e38ad79a2574c9fede9f0cb54d` | Match |
| Participant 08 small archive, Figshare file `53386823` | 241,988,322 bytes | `4666761a184a7c4b019cc78917b75460` | Match |

The Participant 08 archive contains 10 front MP4 files and the corresponding 10 left-view
MP4 files. This archive is a sample of raw videos, not the complete signer data.

A range-read of the Participant 02 large archive central directory confirmed the primary
Figshare representation uses image-frame directories such as
`Participant_02/front/0000/00001.jpg`. The parsed directory had 222,067 entries, including
208,650 `.jpg` files, and all provisional hospital intent IDs listed below were present.

## Pipeline check

The front-view sample `Participant_08/front/0810.mp4`, whose gloss is `出事`, was processed
inside the project Docker image:

```json
{
  "source_frames": 133,
  "valid_frames": 122,
  "valid_ratio": 0.9172932330827067,
  "accepted": true,
  "feature_shape": [48, 368]
}
```

This confirms MP4 decoding, MediaPipe Holistic inference, quality rejection, temporal
resampling, and feature serialization on an official dataset sample. It does not establish
recognition accuracy.

## Hospital vocabulary audit

The following IDs are present in the official 6,707-row gloss table:

| Proposed intent | Dataset ID | Official gloss | Mapping confidence |
|---|---|---|---|
| `REGISTER` | `1660` | 挂号 | Strong |
| `APPOINTMENT` | `1871` | 预约 | Strong |
| `PHARMACY` | `1305` | 药房 | Strong |
| `HELP` | `5094` | 帮助 | Strong |
| `PAIN` | `1682`, `5760` | 疼痛1-1, 疼痛1-2 | Strong, two sign variants |
| `EMERGENCY` | `2465` | 急诊室 | Medium; location is not an emergency declaration |
| `DIRECTIONS` | `4128`, `5604` | 位置2-2, 位置2-1 | Weak; requires semantic review |
| `PAYMENT` | `0533` | 结账 | Weak; hospital payment meaning is not guaranteed |

Additional useful labels include `医院` (`1284`), `医生` (`2102`), `护士` (`5166`),
`检查` (`0811`), `门诊` (`5896`), `病历` (`5790`), `医保卡` (`4323`), and `发票`
(`4384`).

Weak and medium mappings must not be described as linguistically validated. They need review
by a qualified CSL user before the final intent set is frozen.

## Remaining gate

1. Inspect one full image-frame participant archive from `D:\FYP_downloads`.
2. Confirm the exact frame directory structure and selected label IDs.
3. Implement the image-sequence adapter from the observed structure.
4. Benchmark MediaPipe extraction time on at least 20 selected samples.
5. Freeze the label set and signer-independent split only after semantic review.
