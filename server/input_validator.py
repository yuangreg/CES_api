from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict

class InputSetting(BaseModel):
    form: Dict

    @field_validator("form")
    def entry_is_valid(cls, form: Dict) -> List:
        value_list = [
            'longitude',
            'latitude',
            'housing_median_age',
            'total_rooms',
            'total_bedrooms',
            'population',
            'households',
            'median_income'
        ]
        input_list = []
        for val in value_list:
            if val not in form:
                raise ValueError(f"Missing {val} value")
            else:
                try:
                    float_val = float(form[val])
                    input_list.append(float_val)
                except:
                    raise ValueError(f"{val} value is not valid")

        val = 'ocean_proximity'
        if val not in form:
            raise ValueError(f"Missing {val} value")
        else:
            try:
                int_val = int(form[val])
                if int_val > 4:
                    raise ValueError(f"{val} value is not valid")
                vector = [0]*5
                vector[int_val] = 1
                input_list.extend(vector)
            except:
                raise ValueError(f"{val} value is not valid")
        return input_list