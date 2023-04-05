import pandera as pa

# inference output columns
inference_s3_output_columns = pa.DataFrameSchema(
    {
        "CHANNEL_CLASS": pa.Column(pa.String, coerce=True, nullable=True),
        "DISTRIBUTION_CHANNEL": pa.Column(pa.String, coerce=True, nullable=True),
        "DIVISION_CODE": pa.Column(pa.String, coerce=True, nullable=True),
        "SIZE_CODE": pa.Column(pa.String, coerce=True, nullable=True),
        "GENDER_AGE_DESC": pa.Column(pa.String, coerce=True, nullable=True),
        "SILHOUETTE": pa.Column(pa.String, coerce=True, nullable=True),
        "SALES_ORDER_ITEM_VAS_INDICATOR": pa.Column(
            pa.String, coerce=True, nullable=True
        ),
        "VAS_CODE_ZP1": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_SK": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_C20": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_C4X": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_PR": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_C90": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_STD": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_CL1": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_LBC": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_SM": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_CU": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_ES": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_C40": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_CTU": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_CLX": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_SZU": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_REST": pa.Column(pa.String, coerce=True, nullable=True),
        "VAS_CODE_NONE": pa.Column(pa.String, coerce=True, nullable=True),
        "COUNTRY_CODE": pa.Column(pa.String, coerce=True, nullable=True),
        "CUSTOMER_ACCOUNT_GROUP_CODE": pa.Column(pa.String, coerce=True, nullable=True),
        "SALES_ORDER_TYPE": pa.Column(pa.String, coerce=True, nullable=True),
        "AA_INDICATOR": pa.Column(pa.String, coerce=True, nullable=True),
        "FULL_CASE_QUANTITY_MAX": pa.Column(pa.Float, coerce=True, nullable=True),
        "FULL_CASE_QUANTITY_MODE": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOSL_TOTAL_QTY": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOI_TOTAL_QUANTITY": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOH_TOTAL_QUANTITY": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOSL_FULL_CASE_MAX_EQUIVALENT": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "SOSL_FULL_CASE_MODE_EQUIVALENT": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "SOI_FULL_CASE_MAX_EQUIVALENT": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOI_FULL_CASE_MODE_EQUIVALENT": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "SOH_FULL_CASE_MAX_EQUIVALENT": pa.Column(pa.Float, coerce=True, nullable=True),
        "SOH_FULL_CASE_MODE_EQUIVALENT": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "AVG_UNITS_PER_CARTON_SILHOUETTE": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "AVG_UNITS_PER_CARTON_CUSTOMER": pa.Column(
            pa.Float, coerce=True, nullable=True
        ),
        "SALES_ORDER_HEADER_NUMBER": pa.Column(pa.String, coerce=True, nullable=True),
        "SALES_ORDER_ITEM_NUMBER": pa.Column(pa.String, coerce=True, nullable=True),
        "SALES_ORDER_SCHEDULE_LINE_NUMBER": pa.Column(
            pa.String, coerce=True, nullable=True
        ),
        "PREDICTED_CARTONS": pa.Column(pa.Float, coerce=True, nullable=True),
        "INFERENCE_ID": pa.Column(pa.String, coerce=True, nullable=True),
        "PREDICTION_TIMESTAMP": pa.Column(pa.DateTime, nullable=True, coerce=True),
    }
)

inference_sf_output_columns = [
    "sales_order_header_number",
    "sales_order_item_number",
    "sales_order_schedule_line_number",
    "predicted_cartons",
    "prediction_timestamp",
    "creation_date",
]
