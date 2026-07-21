"""为不持有私有根密钥的 CPU 测试提供公开测试 key plan。"""

from __future__ import annotations

from experiments.protocol import formal_randomization


PUBLIC_TEST_ROOT_KEY_MATERIAL = "slm_wm_paper_key"
PUBLIC_TEST_WATERMARK_KEY_PLAN_DIGEST = (
    "2a950250416fd0ec2f3a3aa4c5415e75554d826afd17217074eff8306df12ad3"
)
PUBLIC_TEST_WATERMARK_KEY_PLAN = (
    (
        0,
        8_704_188_491_423_828_510,
        "84a96fe8d4b96a94443b227b134e0dde761be8dc0d23bd1c897fc0945f20aed7",
    ),
    (
        1,
        9_001_591_660_004_305_097,
        "95ab7a3334d9d25add02de5c2f2a6f12061df3ab502d281fd736195583e929a1",
    ),
    (
        2,
        2_607_648_379_522_823_460,
        "4d576053318c518c9031f0d4377c05ea02ca10fcd0bc0933f1cae7a349a78242",
    ),
)


def install_public_test_key_plan() -> None:
    """在 pytest 收集前安装公开测试 plan，不改变仓库生产常量。"""

    formal_randomization.FORMAL_WATERMARK_KEY_PLAN_DIGEST = (
        PUBLIC_TEST_WATERMARK_KEY_PLAN_DIGEST
    )
    formal_randomization.FORMAL_WATERMARK_KEY_PLAN = (
        PUBLIC_TEST_WATERMARK_KEY_PLAN
    )
