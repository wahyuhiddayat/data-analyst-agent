from eval.checker import parse_answers, score_item, values_match


def test_parses_multiple_answers():
    text = "Analysis done.\n@mean_fare[34.65]\n@median_age[28]"
    assert parse_answers(text) == {"mean_fare": "34.65", "median_age": "28"}


def test_keeps_last_value_for_repeated_name():
    text = "@x[1] then corrected: @x[2]"
    assert parse_answers(text) == {"x": "2"}


def test_numeric_match_respects_label_precision():
    assert values_match("34.65", "34.6512")
    assert values_match("0.21", "0.2093")
    assert not values_match("34.65", "34.66")


def test_numeric_match_strips_formatting():
    assert values_match("1200", "1,200")
    assert values_match("34.65", "$34.65")


def test_string_match_is_case_insensitive():
    assert values_match("South", "south")
    assert not values_match("South", "North")


def test_item_correct_requires_all_sub_answers():
    labels = [("mean_fare", "34.65"), ("median_age", "28")]
    result = score_item(labels, "@mean_fare[34.65] @median_age[28.0]")
    assert result["correct"]
    partial = score_item(labels, "@mean_fare[34.65]")
    assert not partial["correct"]
    assert partial["sub_answers_matched"] == 1


def test_positional_fallback_when_names_differ():
    labels = [("correlation_coefficient", "0.21")]
    result = score_item(labels, "@r_value[0.21]")
    assert result["correct"]


def test_dict_value_matches_elementwise():
    expected = "{'month_1': 7.17, 'month_2': 6.53}"
    got = "{'month_1':7.17, 'month_2':6.530}"
    assert values_match(expected, got)
    assert not values_match(expected, "{'month_1': 7.17}")


def test_dict_value_parsed_from_answer_text():
    labels = [("monthly_avg", "{'month_1': 5.9}")]
    result = score_item(labels, "Final: @monthly_avg[{'month_1': 5.90}]")
    assert result["correct"]


def test_nan_matches_nan_only():
    assert values_match("nan", "nan")
    assert values_match("nan", "NaN")
    assert not values_match("nan", "5.0")
    assert not values_match("5.0", "nan")
