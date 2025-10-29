
CREATE TABLE users (
    user_id          VARCHAR(20) PRIMARY KEY,
    userName         VARCHAR(50) NOT NULL,
    userAddress      VARCHAR(100),
    USER_ADR_DETAIL  VARCHAR(100),
    회원_가입일자      DATE
);

CREATE TABLE orders (
    ORDER_ID        BIGINT PRIMARY KEY,
    user_id         VARCHAR(20) NOT NULL,
    order_date      DATE        NOT NULL,
    TOTAL_PRICE     DECIMAL(10, 2)
);

CREATE INDEX user_name_idx ON users (userName);

CREATE FUNCTION CalculateOrderTax(in_total_price DECIMAL)
RETURNS DECIMAL
AS
BEGIN
    RETURN in_total_price * 0.1;
END;