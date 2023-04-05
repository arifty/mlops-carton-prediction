import os
import unittest
from textwrap import dedent

from shapeshifter.inference.input_data_pipeline import InferenceInputDataPipeline


class TestInferenceDataPipeline(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls, *args):
    #     cls.dir_name = os.path.dirname(__file__)
    #     cls.data_pipeline = InferenceInputDataPipeline()

    @classmethod
    def tearDownClass(cls):
        pass

    # def test_get_partition_by_write_columns(self):

    #     output = self.data_pipeline.get_partition_by_write_columns()
    #     expected_output = []
    #     self.assertEqual(output, expected_output)

    # def test_get_query(self):
    #     self.maxDiff = None
    #     test_start_date = "2022-09-01"
    #     test_end_date = "2022-09-30"
    #     output = self.data_pipeline.get_query(test_start_date, test_end_date)
    #     expected_output = f"""
    # WITH FULL_CASE AS (
    #     SELECT
    #         PRODUCT_CODE,
    #         SIZE_CODE,
    #         MAX(CARTON_QUANTITY) AS FULL_CASE_QUANTITY
    #     FROM EMEA_DA_CONS_PROD.INBOUND.SHIPMENT_CARTON_DETAILS_V
    #     --WHERE SHIPMENT_TYPE_CODE='Z001'
    #     GROUP BY
    #         PRODUCT_CODE, SIZE_CODE
    # ),
    # FULL_CASE_BACKUP AS (
    #     SELECT
    #         SILHOUETTE,
    #         AVG(FULL_CASE_QUANTITY) AS FULL_CASE_QUANTITY
    #     FROM (
    #         SELECT
    #             SD.PRODUCT_CODE,
    #             SILHOUETTE,
    #             MAX(CARTON_QUANTITY) AS FULL_CASE_QUANTITY
    #         FROM EMEA_DA_CONS_PROD.INBOUND.SHIPMENT_CARTON_DETAILS_V SD
    #         INNER JOIN EMEA_DA_FDN_PROD.MASTER_DATA.DIM_PRODUCT PROD
    #         ON SD.PRODUCT_CODE = PROD.PRODUCT_CODE
    #             AND PROD.CURRENT_INDICATOR = 1
    #         INNER JOIN EMEA_DA_FDN_PROD.MASTER_DATA.DIM_SILHOUETTE SILH
    #             ON PROD.SK_SILHOUETTE = SILH.SK_SILHOUETTE
    #             AND SILH.CURRENT_INDICATOR = 1
    #         --WHERE SHIPMENT_TYPE_CODE = 'Z001'
    #         GROUP BY
    #             SD.PRODUCT_CODE, SILHOUETTE
    #     )
    #     GROUP BY SILHOUETTE
    # ),
    # VAS AS (
    #     -- one sosl can have multiple vas operations,
    #     -- for that reason we are encoding the vas_packg_cd to boolean variables
    #     SELECT
    #         VAS.SO_DOC_HDR_NBR,
    #         VAS.SO_LN_ITM_NBR,
    #         VAS.SO_SCHED_LN_NBR,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'ZP1', 1, 0)) AS VAS_CODE_ZP1,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'SK', 1, 0)) AS VAS_CODE_SK,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'C20', 1, 0)) AS VAS_CODE_C20,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'C4X', 1, 0)) AS VAS_CODE_C4X,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'PR', 1, 0)) AS VAS_CODE_PR,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'C90', 1, 0)) AS VAS_CODE_C90,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'STD', 1, 0)) AS VAS_CODE_STD,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'CL1', 1, 0)) AS VAS_CODE_CL1,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'LBC', 1, 0)) AS VAS_CODE_LBC,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'SM', 1, 0)) AS VAS_CODE_SM,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'CU', 1, 0)) AS VAS_CODE_CU,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'ES', 1, 0)) AS VAS_CODE_ES,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'C40', 1, 0)) AS VAS_CODE_C40,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'CTU', 1, 0)) AS VAS_CODE_CTU,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'CLX', 1, 0)) AS VAS_CODE_CLX,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) = 'SZU', 1, 0)) AS VAS_CODE_SZU,
    #         MAX(IFF(TRIM(VAS.VAS_PACKG_CD) NOT IN (
    #             'ZP1', 'SK', 'C20', 'C4X',
    #             'PR', 'C90', 'STD', 'CL1',
    #             'LBC', 'SM', 'CU', 'ES',
    #             'C40', 'CTU', 'CLX', 'SZU'
    #         ) AND VAS.VAS_PACKG_CD IS NOT NULL, 1, 0)) AS VAS_CODE_REST,
    #         MAX(IFF(VAS.VAS_PACKG_CD IS NULL, 1, 0)) AS VAS_CODE_NONE
    #     FROM EMEA_DA_CONS_PROD.MPO.VALUE_ADDED_SERVICES_V VAS
    #     GROUP BY
    #         VAS.SO_DOC_HDR_NBR,
    #         VAS.SO_LN_ITM_NBR,
    #         VAS.SO_SCHED_LN_NBR
    # ),
    # AVG_PER_SILHOUETTE AS (
    #     SELECT
    #         SILHOUETTE,
    #         SO_HDR_CRT_DATE AS SALES_ORDER_HEADER_DOCUMENT_DATE,
    #         -- UNBOUNDED CAN BE REPLACED BY A NUMBER TO INDICATE HOW MANY DAYS WE WANT TO USE IN THE SUM
    #         SUM(TOTAL_UNITS) over (order by SO_HDR_CRT_DATE asc rows between unbounded preceding and current row) /
    #             SUM(TOTAL_CARTONS) over (order by SO_HDR_CRT_DATE asc rows between unbounded preceding and current row) AS AVG_UNITS_PER_CARTON_SILHOUETTE
    #     FROM (
    #         SELECT
    #             OBF.SILH_DESC AS SILHOUETTE,
    #             OBF.SO_HDR_CRT_DATE,
    #             COUNT(DISTINCT CI.CARTON_NUMBER) AS TOTAL_CARTONS,
    #             SUM(CI.ACTUAL_DELIVERY_UNITS_QUANTITY) AS TOTAL_UNITS
    #         FROM EMEA_DA_FDN_PROD.DCDASH.CONSUMABLE_CARTON_ITEM_FULL CI
    #         INNER JOIN EMEA_DA_CONS_PROD.DCDASH.VIEW_CONSUMABLE_OUTBOUND_DELIVERY_DOC_DETAIL DD
    #             ON CI.OUTBOUND_DELIVERY_HEADER_NUMBER = DD.OUTBOUND_DELIVERY_HEADER_NUMBER
    #             AND CI.PRODUCT_CODE = DD.PRODUCT_CODE
    #             AND CI.SIZE_CODE = DD.SIZE_CODE
    #         INNER JOIN (SELECT SO_DOC_HDR_NBR, SILH_DESC, SO_HDR_CRT_DATE FROM EMEA_DA_FDN_PROD.MPO.ORDERBOOK_FOUNDATION_HISTORY GROUP BY SO_DOC_HDR_NBR, SILH_DESC, SO_HDR_CRT_DATE) OBF
    #             ON DD.SALES_ORDER_NUMBER = OBF.SO_DOC_HDR_NBR
    #         GROUP BY
    #             OBF.SILH_DESC,
    #             OBF.SO_HDR_CRT_DATE
    #     ) A
    #     WHERE SO_HDR_CRT_DATE >= '2021-01-01'
    # ),
    # AVG_PER_CUSTOMER AS (
    #     SELECT
    #         SHIP_TO_CUSTOMER_NUMBER,
    #         SO_HDR_CRT_DATE AS SALES_ORDER_HEADER_DOCUMENT_DATE,
    #         -- UNBOUNDED CAN BE REPLACED BY A NUMBER TO INDICATE HOW MANY DAYS WE WANT TO USE IN THE SUM
    #         SUM(TOTAL_UNITS) over (order by SO_HDR_CRT_DATE asc rows between unbounded preceding and current row) /
    #             SUM(TOTAL_CARTONS) over (order by SO_HDR_CRT_DATE asc rows between unbounded preceding and current row) AS AVG_UNITS_PER_CARTON_CUSTOMER
    #     FROM (
    #         SELECT
    #             CI.SHIP_TO_CUSTOMER_NUMBER,
    #             OBF.SO_HDR_CRT_DATE,
    #             COUNT(DISTINCT CARTON_NUMBER) AS TOTAL_CARTONS,
    #             SUM(ACTUAL_DELIVERY_UNITS_QUANTITY) AS TOTAL_UNITS
    #         FROM EMEA_DA_FDN_PROD.DCDASH.CONSUMABLE_CARTON_ITEM_FULL CI
    #         INNER JOIN EMEA_DA_CONS_PROD.DCDASH.VIEW_CONSUMABLE_OUTBOUND_DELIVERY_DOC_DETAIL DD
    #             ON CI.OUTBOUND_DELIVERY_HEADER_NUMBER = DD.OUTBOUND_DELIVERY_HEADER_NUMBER
    #             AND CI.PRODUCT_CODE = DD.PRODUCT_CODE
    #             AND CI.SIZE_CODE = DD.SIZE_CODE
    #         INNER JOIN (SELECT SO_DOC_HDR_NBR, SO_HDR_CRT_DATE FROM EMEA_DA_FDN_PROD.MPO.ORDERBOOK_FOUNDATION_HISTORY GROUP BY SO_DOC_HDR_NBR, SO_HDR_CRT_DATE) OBF
    #             ON DD.SALES_ORDER_NUMBER = OBF.SO_DOC_HDR_NBR
    #         WHERE CI.SHIP_TO_CUSTOMER_NUMBER IS NOT NULL
    #         GROUP BY CI.SHIP_TO_CUSTOMER_NUMBER, OBF.SO_HDR_CRT_DATE
    #     ) A
    #     WHERE SO_HDR_CRT_DATE >= '2021-01-01'
    # )
    # SELECT
    #     A.SALES_ORDER_HEADER_NUMBER,
    #     A.SALES_ORDER_ITEM_NUMBER,
    #     A.SALES_ORDER_SCHEDULE_LINE_NUMBER,
    #     A.SHIP_TO_CUSTOMER_NUMBER,
    #     A.CHANNEL_CLASS,
    #     A.DISTRIBUTION_CHANNEL,
    #     A.DIVISION_CODE,
    #     A.PRODUCT_CODE,
    #     A.SIZE_CODE,
    #     A.GENDER_AGE_DESC,
    #     A.SILHOUETTE,
    #     A.SALES_ORDER_ITEM_VAS_INDICATOR,
    #     A.VAS_CODE_ZP1,
    #     A.VAS_CODE_SK,
    #     A.VAS_CODE_C20,
    #     A.VAS_CODE_C4X,
    #     A.VAS_CODE_PR,
    #     A.VAS_CODE_C90,
    #     A.VAS_CODE_STD,
    #     A.VAS_CODE_CL1,
    #     A.VAS_CODE_LBC,
    #     A.VAS_CODE_SM,
    #     A.VAS_CODE_CU,
    #     A.VAS_CODE_ES,
    #     A.VAS_CODE_C40,
    #     A.VAS_CODE_CTU,
    #     A.VAS_CODE_CLX,
    #     A.VAS_CODE_SZU,
    #     A.VAS_CODE_REST,
    #     A.VAS_CODE_NONE,
    #     A.SHIPPING_LOCATION_CODE,
    #     A.COUNTRY_CODE,
    #     A.CUSTOMER_ACCOUNT_GROUP_CODE,
    #     A.SALES_ORDER_TYPE,
    #     A.AA_INDICATOR,
    #     MAX(A.FULL_CASE_QUANTITY) AS FULL_CASE_QUANTITY,
    #     MAX(A.ORDER_SCHEDULE_LINE_QTY) AS SOSL_TOTAL_QTY,
    #     -- Since we added delivery docs we have to 'deduplicate' the quantities here to get the correct number on item and header level
    #     SUM(MAX(A.CNFRMD_QTY)) OVER (PARTITION BY A.SALES_ORDER_HEADER_NUMBER, A.SALES_ORDER_ITEM_NUMBER) AS SOI_TOTAL_QUANTITY,
    #     SUM(MAX(A.CNFRMD_QTY)) OVER (PARTITION BY A.SALES_ORDER_HEADER_NUMBER) AS SOH_TOTAL_QUANTITY,
    #     MAX(A.ORDER_SCHEDULE_LINE_QTY) / MAX(A.FULL_CASE_QUANTITY) AS SOSL_FULL_CASE_EQUIVALENT,
    #     SUM(MAX(A.CNFRMD_QTY)) OVER (PARTITION BY A.SALES_ORDER_HEADER_NUMBER, A.SALES_ORDER_ITEM_NUMBER) / MAX(A.FULL_CASE_QUANTITY) AS SOI_FULL_CASE_EQUIVALENT,
    #     SUM(MAX(A.CNFRMD_QTY)) OVER (PARTITION BY A.SALES_ORDER_HEADER_NUMBER) / MAX(A.FULL_CASE_QUANTITY) AS SOH_FULL_CASE_EQUIVALENT,
    #     -- !!! APPLY FORWARD FILL IN PYTHON FOR AVG_UNITS_PER_CARTON_SILHOUETTE
    #     MAX(AVG_PER_SILHOUETTE.AVG_UNITS_PER_CARTON_SILHOUETTE) AS AVG_UNITS_PER_CARTON_SILHOUETTE,
    #     -- !!! APPLY FORWARD FILL IN PYTHON FOR AVG_UNITS_PER_CARTON_CUSTOMER
    #     MAX(AVG_PER_CUSTOMER.AVG_UNITS_PER_CARTON_CUSTOMER) AS AVG_UNITS_PER_CARTON_CUSTOMER,
    #     A.SALES_ORDER_HEADER_DOCUMENT_DATE
    # FROM (
    #     SELECT
    #         OBF.SO_DOC_HDR_NBR AS SALES_ORDER_HEADER_NUMBER,
    #         OBF.SO_LN_ITM_NBR AS SALES_ORDER_ITEM_NUMBER,
    #         OBF.SO_SCHED_LN_NBR AS SALES_ORDER_SCHEDULE_LINE_NUMBER,
    #         OBF.CNFRMD_QTY,
    #         OBF.CUST_SHIP_TO_CD AS SHIP_TO_CUSTOMER_NUMBER,
    #         OBF.CNFRMD_QTY AS ORDER_SCHEDULE_LINE_QTY,
    #         OBF.DISTRIB_MTHD_CD AS DISTRIBUTION_CHANNEL,
    #         OBF.CUST_CHNL_DESC AS CHANNEL_CLASS,
    #         OBF.DIV_CD AS DIVISION_CODE,
    #         OBF.PROD_CD AS PRODUCT_CODE,
    #         OBF.SZ_DESC AS SIZE_CODE,
    #         OBF.GNDR_AGE_DESC AS GENDER_AGE_DESC,
    #         COALESCE(FULL_CASE.FULL_CASE_QUANTITY, FULL_CASE_BACKUP.FULL_CASE_QUANTITY) AS FULL_CASE_QUANTITY,
    #         OBF.SILH_DESC AS SILHOUETTE,
    #         OBF.SO_ITM_VAS_IND AS SALES_ORDER_ITEM_VAS_INDICATOR,
    #         COALESCE(VAS_CODE_ZP1, 0) AS VAS_CODE_ZP1,
    #         COALESCE(VAS_CODE_SK, 0) AS VAS_CODE_SK,
    #         COALESCE(VAS_CODE_C20, 0) AS VAS_CODE_C20,
    #         COALESCE(VAS_CODE_C4X, 0) AS VAS_CODE_C4X,
    #         COALESCE(VAS_CODE_PR, 0) AS VAS_CODE_PR,
    #         COALESCE(VAS_CODE_C90, 0) AS VAS_CODE_C90,
    #         COALESCE(VAS_CODE_STD, 0) AS VAS_CODE_STD,
    #         COALESCE(VAS_CODE_CL1, 0) AS VAS_CODE_CL1,
    #         COALESCE(VAS_CODE_LBC, 0) AS VAS_CODE_LBC,
    #         COALESCE(VAS_CODE_SM, 0) AS VAS_CODE_SM,
    #         COALESCE(VAS_CODE_CU, 0) AS VAS_CODE_CU,
    #         COALESCE(VAS_CODE_ES, 0) AS VAS_CODE_ES,
    #         COALESCE(VAS_CODE_C40, 0) AS VAS_CODE_C40,
    #         COALESCE(VAS_CODE_CTU, 0) AS VAS_CODE_CTU,
    #         COALESCE(VAS_CODE_CLX, 0) AS VAS_CODE_CLX,
    #         COALESCE(VAS_CODE_SZU, 0) AS VAS_CODE_SZU,
    #         COALESCE(VAS_CODE_REST, 0) AS VAS_CODE_REST,
    #         COALESCE(VAS_CODE_NONE, 0) AS VAS_CODE_NONE,
    #         OBF.SHPG_LCTN_CD AS SHIPPING_LOCATION_CODE,
    #         OBF.SO_TYPE_DESC AS SALES_ORDER_TYPE,
    #         CUST.COUNTRY_CODE,
    #         CUST.CUSTOMER_ACCOUNT_GROUP_CODE,
    #         OBF.AA_IND AS AA_INDICATOR,
    #         SO_HDR_CRT_DATE AS SALES_ORDER_HEADER_DOCUMENT_DATE
    #     FROM EMEA_DA_FDN_PROD.MPO.ORDERBOOK_FOUNDATION OBF
    #     LEFT JOIN FULL_CASE
    #         ON FULL_CASE.PRODUCT_CODE = OBF.PROD_CD
    #         AND FULL_CASE.SIZE_CODE = OBF.SZ_DESC
    #     LEFT JOIN FULL_CASE_BACKUP
    #         ON FULL_CASE_BACKUP.SILHOUETTE = OBF.SILH_DESC
    #     LEFT JOIN EMEA_DA_FDN_PROD.MASTER_DATA.DIM_CUSTOMER_MASTER CUST
    #         ON CUST.CUSTOMER_NUMBER = OBF.CUST_SHIP_TO_CD
    #         AND CUST.CURRENT_INDICATOR = 1
    #     LEFT JOIN VAS
    #         ON VAS.SO_DOC_HDR_NBR = OBF.SO_DOC_HDR_NBR
    #         AND VAS.SO_LN_ITM_NBR = OBF.SO_LN_ITM_NBR
    #         AND VAS.SO_SCHED_LN_NBR = OBF.SO_SCHED_LN_NBR
    #     WHERE   OBF.ACTVIND='Y'
    #         AND OBF.SO_HDRACTVIND='Y'
    #         AND OBF.SO_ITMACTVIND='Y'
    #         AND OBF.SO_SCHEDACTVIND='Y'
    #         AND OBF.SO_TYPE_CD IN ('Z001','Z003','Z004','Z005','Z011','Z014','Z020','Z030')
    #         AND OBF.RMAING_TO_SHIP_RPTG_CNFRMD_QTY > 0
    #         -- AND OBF.SO_IDP_IND = 'X' -- NEEDED, I am checking with Shenyang??
    #         AND (
    #             CASE  WHEN OBF.DLVRY_TYPE_IND = 'O'
    #             THEN COALESCE(OBF.TOT_SHPD_QTY,0)
    #             ELSE 0 end = 0 OR
    #             CASE WHEN OBF.DLVRY_TYPE_IND = 'I'
    #             THEN COALESCE(OBF.TOT_SHPD_QTY,0)
    #             ELSE 0 end = 0
    #         )
    # ) A
    # LEFT JOIN AVG_PER_SILHOUETTE
    #     ON AVG_PER_SILHOUETTE.SILHOUETTE = A.SILHOUETTE
    #     AND AVG_PER_SILHOUETTE.SALES_ORDER_HEADER_DOCUMENT_DATE = A.SALES_ORDER_HEADER_DOCUMENT_DATE
    # LEFT JOIN AVG_PER_CUSTOMER
    #     ON AVG_PER_CUSTOMER.SHIP_TO_CUSTOMER_NUMBER = A.SHIP_TO_CUSTOMER_NUMBER
    #     AND AVG_PER_CUSTOMER.SALES_ORDER_HEADER_DOCUMENT_DATE = A.SALES_ORDER_HEADER_DOCUMENT_DATE
    # GROUP BY
    #     A.SALES_ORDER_HEADER_NUMBER,
    #     A.SALES_ORDER_ITEM_NUMBER,
    #     A.SALES_ORDER_SCHEDULE_LINE_NUMBER,
    #     A.SHIP_TO_CUSTOMER_NUMBER,
    #     A.CHANNEL_CLASS,
    #     A.DISTRIBUTION_CHANNEL,
    #     A.DIVISION_CODE,
    #     A.PRODUCT_CODE,
    #     A.SIZE_CODE,
    #     A.GENDER_AGE_DESC,
    #     A.SILHOUETTE,
    #     A.SALES_ORDER_ITEM_VAS_INDICATOR,
    #     A.VAS_CODE_ZP1,
    #     A.VAS_CODE_SK,
    #     A.VAS_CODE_C20,
    #     A.VAS_CODE_C4X,
    #     A.VAS_CODE_PR,
    #     A.VAS_CODE_C90,
    #     A.VAS_CODE_STD,
    #     A.VAS_CODE_CL1,
    #     A.VAS_CODE_LBC,
    #     A.VAS_CODE_SM,
    #     A.VAS_CODE_CU,
    #     A.VAS_CODE_ES,
    #     A.VAS_CODE_C40,
    #     A.VAS_CODE_CTU,
    #     A.VAS_CODE_CLX,
    #     A.VAS_CODE_SZU,
    #     A.VAS_CODE_REST,
    #     A.VAS_CODE_NONE,
    #     A.SHIPPING_LOCATION_CODE,
    #     A.COUNTRY_CODE,
    #     A.CUSTOMER_ACCOUNT_GROUP_CODE,
    #     A.SALES_ORDER_TYPE,
    #     A.AA_INDICATOR,
    #     A.SALES_ORDER_HEADER_DOCUMENT_DATE
    # ;
    #     """

    #     self.assertEqual(
    #         dedent(" ".join("".join(output.splitlines()).split())),
    #         dedent(" ".join("".join(expected_output.splitlines()).split())),
    #     )
