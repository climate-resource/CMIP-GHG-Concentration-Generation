"""
Creation of configuration for handling AGAGE's data
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig

DOWNLOAD_URLS = {
    ("ch4", "gc-md", "monthly"): [
        URLSource(
            known_hash="e6c3955c0e9178333c5f2177088a9fe84ec27b557901364750a82241f3477300",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_ch4_mon.txt",
        ),
        URLSource(
            known_hash="91cbef846e4158a880515b3b86b5b28d7510dcc6cf9494e3fec823e0c3f0678c",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="3d295bad0b883b6099ed5171044ed7a46e5ae93e8646a2020058a72c648ed0a6",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_ch4_mon.txt",
        ),
        URLSource(
            known_hash="e775e79fcf6cb833aa7d139c79725f25aefb81d4e90557616c4939d497f80719",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="fceb3a14534ce94d550f24831c7fc1258700f24b1a917005b6c06a85843ce0e1",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_ch4_mon.txt",
        ),
    ],
    ("n2o", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_n2o_mon.txt",
            known_hash="7ac04ee39e56544dc6d98e68d52be58ab30cde620f511d70192a005c95b85fc0",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_n2o_mon.txt",
            known_hash="7a0d7481d6d5492bf4501c107c57c6f6b0bb9991eb9ff23abb8c9c966fefa79c",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_n2o_mon.txt",
            known_hash="849efb23da2bfad5e76e5abd7303bb3b74b3bd9f924daa4303bbc10d04cf67da",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_n2o_mon.txt",
            known_hash="3739081cf7778e205dc1c77c74ffdb72d748baaf346858850fd39e41f16f42c3",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_n2o_mon.txt",
            known_hash="e4cdc474f71d6f80ac63fe13b7fa61a86ce0361b7995874bf074a01c69facac3",
        ),
    ],
    ("cfc11", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_cfc-11_mon.txt",
            known_hash="7f52297786487e9ede8a785c89b1d793250c4198223bafaa265cdcc268bbc978",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_cfc-11_mon.txt",
            known_hash="ebc838159a6ccc0bb98ac16203e927abea3371aa8aea801db572c816761074cd",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_cfc-11_mon.txt",
            known_hash="aa6ef6875861e0d127fadc4697467b08ba8272d0fd3be91dd63713490de86645",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_cfc-11_mon.txt",
            known_hash="d1722aa15bf3a77415b97f0f9a1c1e58912e0ace054b00c8767e75666f877318",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_cfc-11_mon.txt",
            known_hash="9929b38d196ef1397615b51988631f1e790797d86c260f57b387074cb667ef56",
        ),
    ],
    ("cfc11", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-11_mon.txt",
            known_hash="6d8b653daf80d3fd8c295c91e8842e8c81242c36a543da88b19964c1de7ef7ad",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-11_mon.txt",
            known_hash="1f2dd4650b49c7ee5d9e5a763e1be3daeb72f0243ea249ab3b9c9d586e71f8c4",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-11_mon.txt",
            known_hash="35231a8b2a6776e815c843ad7ba0f99378733dbbaa6c112f33d30d0558e66ad8",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_cfc-11_mon.txt",
            known_hash="33fbb30c985d2ae36a48f5a7e6e92e66ea84c83004db006ca2fb1de72f922112",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-11_mon.txt",
            known_hash="f1eb53ecfa4294ef3581b536804fccac36519dbc4ddafa856c4c4aeb9e7aa048",
        ),
    ],
    ("cfc11", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_cfc-11_mon.txt",
            known_hash="3394d57cc17222ccc5de8f98edbe26131afc63e718ff9d4563727a098704aa93",
        )
    ],
    ("cfc113", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_cfc-113_mon.txt",
            known_hash="74610a0d9de3dbbbb58ff9d665f9bee799882d2b65648a6ccc36b964b145a12a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_cfc-113_mon.txt",
            known_hash="3572e7c19ca325751244a0072cf06f5cec97edea41f584bbc410b52a64b1aa2f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_cfc-113_mon.txt",
            known_hash="c1979c4b61673019ccbc6801b5c13b29fa867cd52ef64c4d964cf5ff1a21998d",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_cfc-113_mon.txt",
            known_hash="36bbd53c1d932742e8cfe0cd6d3e4cf31f141e23eb99bf14ee8e1f091db76c01",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_cfc-113_mon.txt",
            known_hash="82043f237f083079c7f15f28449a1f2fe397416cce8f89a061ddd6e7c6405bce",
        ),
    ],
    ("cfc113", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_cfc-113_mon.txt",
            known_hash="98c7e141bf65b050cf4f9a6b2b602fba5bf2ca17eca5c80e51beb2ece26ac29a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_cfc-113_mon.txt",
            known_hash="d8e071d1758cce17b6cbf85e81307b56559dba5bfc7cc2b2bda3f88886d724e9",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-113_mon.txt",
            known_hash="c45e5c75f7b3054412e34d1aee0f15200dccddabc42fb1a2ee09e5ff0690d0be",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-113_mon.txt",
            known_hash="302c34bcdf5f5dbc65cd922499af65d66099249660b0444cdd79f8c2f7eb1ba5",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_cfc-113_mon.txt",
            known_hash="4a982d249b179c336be985f8610a394e661c4e3d278dbf986571fe71f6787499",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-113_mon.txt",
            known_hash="90379724f71bb19546c5f9bdc997d56b37af47fc302acc0882c3e00888bcca8f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_cfc-113_mon.txt",
            known_hash="9edbd8c6789ec71deb553facf132e8d0eb99ec5c7d7fbf39827047a646c63148",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_cfc-113_mon.txt",
            known_hash="8cfae7483c4b3750eae1701d78bd6ce23ef5fc26ffb81ec1dd158118ca08acad",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_cfc-113_mon.txt",
            known_hash="ee374631a4a9e76f633ff527da2abbe5ac2ea3f83c1c2f6dbd8351ae469ee06e",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-113_mon.txt",
            known_hash="5b77bb90ed6cabf5e502ea5acd6fd5b5fda0015d51e116c835903ef0234a565a",
        ),
    ],
    ("cfc12", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_cfc-12_mon.txt",
            known_hash="504f4d3db1bea293392d3efaf151bfad7e5f2277224cc765877e00e69fd02be0",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_cfc-12_mon.txt",
            known_hash="01b70bb4ba8abd98c02f07259c27ce8a105d4208e7bf423f406e4b8da6b480f3",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_cfc-12_mon.txt",
            known_hash="400e82052314190309650b596a1e2e8bb6ed06e1bff67d831df8b5e1d5738d1b",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_cfc-12_mon.txt",
            known_hash="28823f27e07f83db07b2507f038cf046bf51697bca8cc19e7b50414e1aea2775",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_cfc-12_mon.txt",
            known_hash="737e826ab87381fd98578363c5bd040d92c33356ede9af3aae802824a47887c9",
        ),
    ],
    ("cfc12", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-12_mon.txt",
            known_hash="50344532103b093e6442f8f329a4bdb41c578a72f1be77836bb3170b20abab57",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-12_mon.txt",
            known_hash="fa7ac9be9f3d650d4cbd5840b7905968f28a6a6a88a8f5e43ea06fa0f1f29ac2",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-12_mon.txt",
            known_hash="f46822e1c50c6b51db99066ffa671548709d6455fcb6b3989f75643383b49bab",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_cfc-12_mon.txt",
            known_hash="5db4ff4ce96cc261c386a62d9d1b59a8458d8db2edc2db7c5b62f6e686ba2989",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-12_mon.txt",
            known_hash="d018493146fcee2f9750275dd561dd74ff827c207c28435727ba1bb9164e07d2",
        ),
    ],
    ("cfc12", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_cfc-12_mon.txt",
            known_hash="0325346db119008b4e8a7dd7b9641b8eb097b6b03e69e6af29f240155ace3a2e",
        )
    ],
    ("cfc114", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_cfc-114_mon.txt",
            known_hash="10d38164cbc335b50cb090396077bf8f9b449f176fc770416b127eb682546d3f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_cfc-114_mon.txt",
            known_hash="b544530b8e0e42159b890761ec6e930401ef641176f4b10e30adf9254c5e4083",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-114_mon.txt",
            known_hash="4a5703f6b4318efff2a094fb7e8499ad46e250a8b9f25c8c5d4950fcf16ddf5a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-114_mon.txt",
            known_hash="ec5ceaf9aa9c35161f3f58c380a3b28ecc7cce3ca2e3aab710b792b3008c6d1f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_cfc-114_mon.txt",
            known_hash="5decb9819446deefcde82198c7f12b3588e2c3c20e9f25680335df1558e815e4",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-114_mon.txt",
            known_hash="31efeb79fd9ae410414409d00038ea94d1a6c50829dbc5b699b49d4f7b1f2e7d",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_cfc-114_mon.txt",
            known_hash="503e67b89d6464a8ad1e22b66fa943ec5d7be6f6a3f6d5699931cc2b2c93d4ec",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_cfc-114_mon.txt",
            known_hash="08124810bd93073048ae9bf0d62399fb5eba0932082c2821309146c32ecf14ec",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_cfc-114_mon.txt",
            known_hash="39b531a6c52e78bd7ff7fd5ec097b54ed64044487a7b015075081098ee2e171f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-114_mon.txt",
            known_hash="4a2ddb57bb738ed4abf8c667caea31e61c78778a03de3a2a79aa5f9b658b91ca",
        ),
    ],
    ("cfc114", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_cfc-114_mon.txt",
            known_hash="8b0e85a59515871ae246d580e29f4876de3289cf9617a857fa3bcb045afb456a",
        )
    ],
    ("cfc115", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_cfc-115_mon.txt",
            known_hash="af8ad4291e67bbfb150aef9c9c654734c048df72e2bf3297bb1873a2c4b747ad",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_cfc-115_mon.txt",
            known_hash="a5f686837b632436315c93591a476026a533a7a3932558c510731b4e159aa181",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-115_mon.txt",
            known_hash="f4d2b5470ef6dc044a8708a422676f688b80f9efa68ae8ccad6577ac41afb9ee",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-115_mon.txt",
            known_hash="01ed606b0d6031d93164b958160a9f9a7bb0c673abc77b2357253fa5b7524c54",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_cfc-115_mon.txt",
            known_hash="ed8e8830d92434172948da6855be95112e4741330209cf6e831afc14e7ccf0bc",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-115_mon.txt",
            known_hash="ba8306df96f687f5aad5b43d3016344e0ba9226229e6f6d12a8a0fc90acc74a8",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_cfc-115_mon.txt",
            known_hash="1c1691c4c3e71f039b3bda4e3ab9b9f834f42c42a42e63a5f6997a9f2c00c511",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_cfc-115_mon.txt",
            known_hash="5c3a34be2dcc440c419e0cb8fa376c950a0379597c4354a3af05c3f959647c6f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-115_mon.txt",
            known_hash="ee52cc6800e535186e1c49890ff12fe34169339538e38dbc6daa3ec09cd9e6a4",
        ),
    ],
    ("cfc115", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_cfc-115_mon.txt",
            known_hash="f33237c481f69babfe4f1dc848db048121d7ef7c8899da8efd9e7f88f116a448",
        )
    ],
    ("ch2cl2", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_ch2cl2_mon.txt",
            known_hash="59810385019e5e959037ea7aa760c4ab676162b8b0dbc562c016a65d94121d64",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_ch2cl2_mon.txt",
            known_hash="7bf38ab80ff3802c9180da0d3e3ee531c8104696d2ad090d21aa83d74d88ea42",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_ch2cl2_mon.txt",
            known_hash="af9c257b7d994839968ae84af893c15090fa9191f40c7f54961b417d8b8fe538",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_ch2cl2_mon.txt",
            known_hash="f8dfe808bc625d52a1b6c2eca26e6a86b60492c32c7b0221d7bdd20e6ccb944b",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_ch2cl2_mon.txt",
            known_hash="1b6368523ea1de206d00446b0b55d685b447b6eeb65930d0219d4983d1796a21",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_ch2cl2_mon.txt",
            known_hash="d44a0a926741d96514266620bf25be3a738bbc3dc2915ca87c93ce8df00fff02",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_ch2cl2_mon.txt",
            known_hash="450764ffa7057f162a8a7204359d8bbd7c59c6d9b8f727092465f7c01101be11",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_ch2cl2_mon.txt",
            known_hash="8598743c2a1ce6b0b0beebd30b5959d7bf46104bd962c3326e5137d41b47cd97",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_ch2cl2_mon.txt",
            known_hash="16c93d3985d6369de2f02265b707f475a51c7dbe2c4bd9de95ca991e501b6f90",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_ch2cl2_mon.txt",
            known_hash="640cf7b267af3925dc912f9e592a5d9c52e51c433fe3ecc25171e2512f89e0c1",
        ),
    ],
    ("ch2cl2", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/capegrim/ascii/AGAGE-GCMS-ADS_CGO_ch2cl2_mon.txt",
            known_hash="9aa458b1b1d175e646a411d23a4499ec8bcc78945d2729adcc1d5758c389142c",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/jungfraujoch/ascii/AGAGE-GCMS-ADS_JFJ_ch2cl2_mon.txt",
            known_hash="2ced9bf1c4eee47f251e83166055a51fd3fb8870cecf69fcfcb6c21bc37132cf",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/macehead/ascii/AGAGE-GCMS-ADS_MHD_ch2cl2_mon.txt",
            known_hash="4b0e27ed9a40b1fd7bc1d25f5906ac1bf8e27f9be1ecd7cd4c91a418d8437c3d",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_ch2cl2_mon.txt",
            known_hash="e8afe5118c2e619c7da121694e2c47da2e3014784d7b9467c61085441c96eb90",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/zeppelin/ascii/AGAGE-GCMS-ADS_ZEP_ch2cl2_mon.txt",
            known_hash="5c5433c25666e266ee2992ee2be17f5631e830e4be4ddc2516b1b82aa53cb3ce",
        ),
    ],
    # ("cfc11", "gc-md", "monthly"): [
    #
    # ],
    # ("cfc11", "gc-ms-medusa", "monthly"): [
    #
    # ],
    # ("cfc11", "gc-ms", "monthly"): [
    #
    # ],
    ("hfc134a", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_hfc-134a_mon.txt",
            known_hash="d2f39aa42e4ae6182084d595bb51d7a928913819ec30dedc5fab2b09ebce50fa",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_hfc-134a_mon.txt",
            known_hash="289fd88e1f6e8aa0fce15dadc1e9c1d246108d9a9d615327c4e242dbbbf8095c",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_hfc-134a_mon.txt",
            known_hash="bfdb55db913a285192ac5cdb9456d452cf75d190f5df2d7ad58b10b91bb5211b",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_hfc-134a_mon.txt",
            known_hash="7078f3636bbd582f09f9706c4a36bd07a3ea2194800d4dc485a01ac6cefbd3be",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_hfc-134a_mon.txt",
            known_hash="b30dc3b0829d52cda6554a63558f20e2e23713562e6d08cc28beeecb770b9dbe",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_hfc-134a_mon.txt",
            known_hash="67bcdb81b91af3a417cf60a7b3fad1c3feb7e6b1ba53ba509c75da9c67754d03",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_hfc-134a_mon.txt",
            known_hash="3aefaadbb40df63585397d3fe8ef4f5ce9980264bd68d2f281d7f3e40177267a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_hfc-134a_mon.txt",
            known_hash="31c4b1d2c15cfe77869a72ee60f6e749fbb775dc5a62451637e72097c3cd0d21",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_hfc-134a_mon.txt",
            known_hash="b61be1912adf36961a37127f78daa42f903b044e796493c7a46ae180c036aa72",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_hfc-134a_mon.txt",
            known_hash="724b699558e9c26cccaff47ef18ce73812cc1a0b25fdb3a2e93d01893f0c564d",
        ),
    ],
    ("hfc134a", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/capegrim/ascii/AGAGE-GCMS-ADS_CGO_hfc-134a_mon.txt",
            known_hash="8c6405431ea194670ac50816bf7a01556bf240d95110098c6f8203a27c73eefd",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/jungfraujoch/ascii/AGAGE-GCMS-ADS_JFJ_hfc-134a_mon.txt",
            known_hash="7b0a36cc62629440de2d4ed11c4a52744d38bdac0dc096ee35144bced8ebb032",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/macehead/ascii/AGAGE-GCMS-ADS_MHD_hfc-134a_mon.txt",
            known_hash="8ffe2a817fe60d8427a88fdad07937d1103d70cf7780cfac409fe20676bdc4b4",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_hfc-134a_mon.txt",
            known_hash="e69c58e0a4f96d9782645d837e8189582e8471ff4ad22b3f9230bd038ccb4600",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/zeppelin/ascii/AGAGE-GCMS-ADS_ZEP_hfc-134a_mon.txt",
            known_hash="2cce94a135d6a8f48dcfd487e9196265c8f290d1f29a0b98ef5020607d615516",
        ),
    ],
    ("sf6", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim-sf6-ecd/ascii/AGAGE-GC-ECD-SF6_CGO_sf6_mon.txt",
            known_hash="9ec255bbd55447d8ac521a4683951e0b0aa682d8252a1072e5fa555b90af5aa1",
        )
    ],
    ("sf6", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_sf6_mon.txt",
            known_hash="1611fb6ed6087b506af19ca8a08cdf750e2c185b136b2460ba013ace39b47714",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_sf6_mon.txt",
            known_hash="d387ab096cc53fae4efa5026eaaba4f3df0ceec7a1afcfc1128687372e6505d3",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_sf6_mon.txt",
            known_hash="b556daf22abd2edf0874b875a60bd02f246315c4c5a9748273362cb210c7e077",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_sf6_mon.txt",
            known_hash="030a37af25e2d806d8ac65be66f264f449923a1b8b077c92c647577d4efe3720",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_sf6_mon.txt",
            known_hash="42d1f57226972df7d16786310e95288db0b604033d8fac82b8b03630d36d908a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_sf6_mon.txt",
            known_hash="cb05383d875d6020a0017942551f909954354ed2701a12754da6bdb80e45612f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_sf6_mon.txt",
            known_hash="95fa2ccc93fe2dfefcb64d1405e81e0bc556a1892e5b4818312003de92eaf23b",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_sf6_mon.txt",
            known_hash="12270c0ab91decaaffc0f47710c27de9147a125938c3e98b1c10a98a2922f441",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_sf6_mon.txt",
            known_hash="da146fc89d0e2b2e87846c6e5fc5712820a5f7bff69f054d90ac0ce80a1cf2a7",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_sf6_mon.txt",
            known_hash="2de77c7f417510878c18d4ea452815ac4555ae17719e6786548b063ee471e5bf",
        ),
    ],
}


def create_agage_handling_config(
    data_sources: tuple[tuple[str, str, str]],
) -> list[RetrieveExtractAGAGEDataConfig]:
    """
    Create config for handling AGAGE data

    Parameters
    ----------
    data_sources
        Data sources to retrieve.
        Each input tuple should contain
        the gas of interest (zeroth element),
        the instrument of interest (first element)
        and the time frequency of interest (second element).

    Returns
    -------
        Configuration for handling AGAGE data for the requested data sources.
    """
    res = []
    for data_source in data_sources:
        gas, instrument, frequency = data_source

        raw_dir = Path("data/raw/agage/agage")
        interim_dir = Path("data/interim/agage/agage")

        res.append(
            RetrieveExtractAGAGEDataConfig(
                step_config_id=f"{gas}_{instrument}_{frequency}",
                gas=gas,
                instrument=instrument,
                time_frequency=frequency,
                raw_dir=raw_dir,
                download_complete_file=raw_dir
                / f"{gas}_{instrument}_{frequency}.complete",
                processed_monthly_data_with_loc_file=interim_dir
                / f"{gas}_{instrument}_{frequency}.csv",
                generate_hashes=False,
                download_urls=DOWNLOAD_URLS[(gas, instrument, frequency)],
            )
        )

    return res
